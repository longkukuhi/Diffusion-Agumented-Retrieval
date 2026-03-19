import json
import os

import torch
import torch.nn.functional as F
import tqdm

from baselines import ImageEmbedder, BLIP_BASELINE


class Corpus(torch.utils.data.Dataset):
    """Dataset class for the corpus images (the 50k potential candidates)."""

    def __init__(self, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        return self.path2id[path]

    def __getitem__(self, i):
        image = self.preprocessor(self.corpus[i])
        return {'id': i, 'image': image}


class GeneratedImageCorpus(torch.utils.data.Dataset):
    """Dataset class for generated images used in reranking."""

    def __init__(self, queries_path, preprocessor, image_path):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.preprocessor = preprocessor
        self.image_root = image_path

    def __len__(self):
        return len(self.queries) * 11

    def __getitem__(self, idx):
        dialog_idx = idx // 11
        round_idx = idx % 11
        img_path = os.path.join(self.image_root, f'{dialog_idx}_{round_idx}.jpg')
        image = self.preprocessor(img_path)
        return {'idx': idx, 'image': image}


class Queries(torch.utils.data.Dataset):
    """Dataset class for query dialogues and their target images."""

    def __init__(self, cfg, queries_path):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.cfg = cfg

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, (
            "Please set self.dialog_length=<DIALOG_LENGTH> to any number in [0, ..., 10]"
        )
        target_path = self.queries[i]['img']
        text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path, 'idx': i}


def build_model(cfg):
    if cfg['use_generated_image']:
        cfg['cache_gen_corpus'] = 'temp/blip_val_generated_images_genimg.pth'
    else:
        cfg['cache_gen_corpus'] = ''

    cfg['cache_corpus'] = 'temp/corpus_blip_small.pth'
    dialog_encoder, image_embedder = BLIP_BASELINE()
    return cfg, dialog_encoder, image_embedder


class ChatIREval:
    """Main evaluation pipeline."""

    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder):
        self.dialog_encoder = dialog_encoder
        self.image_embedder = image_embedder
        self.cfg = cfg
        self.corpus = None
        self.generated_images = None
        self.corpus_dataset = Corpus(self.cfg['corpus_path'], self.image_embedder.processor)

        if self.cfg['use_generated_image']:
            self.generated_img_path = 'val_generated_images'
            self.generated_images_dataset = GeneratedImageCorpus(
                self.cfg['queries_path'],
                self.image_embedder.processor,
                self.generated_img_path,
            )

    def _get_recalls(self, dataloader, dialog_length):
        dataloader.dataset.dialog_length = dialog_length
        recalls = []

        for batch in tqdm.tqdm(dataloader):
            target_ids = torch.tensor(
                [self.corpus_dataset.path_to_index(p) for p in batch['target_path']]
            ).unsqueeze(1).to(self.cfg['device'])
            text_ids = torch.tensor(batch['idx']).unsqueeze(1).to(self.cfg['device'])

            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)

            if self.cfg['use_generated_image']:
                img_idx = text_ids * 11 + dialog_length
                gen_img_feats = self.generated_images[1][img_idx].squeeze(1)

                visual_scores = gen_img_feats @ self.corpus[1].T
                textual_scores = pred_vec @ self.corpus[1].T

                if dialog_length < 2:
                    visual_weight = 0.2
                    text_weight = 0.8
                else:
                    visual_weight = 0.5
                    text_weight = 0.5

                scores = visual_weight * visual_scores + text_weight * textual_scores
            else:
                scores = pred_vec @ self.corpus[1].T

            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)

        return torch.cat(recalls)

    def run(self, hits_at=10):
        assert self.corpus is not None, 'Prepare corpus first via self.index_corpus().'

        dataset = Queries(self.cfg, self.cfg['queries_path'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['queries_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

        hits_results = []
        for dialog_length in range(11):
            print(f'Calculate recalls for dialogues of length {dialog_length}...')
            dialog_recalls = self._get_recalls(dataloader, dialog_length=dialog_length)
            hits_results.append(dialog_recalls)

        hits_results = cumulative_hits_per_round(
            torch.cat(hits_results).cpu(), hitting_recall=hits_at
        ).tolist()

        print(f'====== Results for Hits@{hits_at} ======')
        for dialog_length in range(11):
            print(f'\tDialog Length: {dialog_length}: {round(hits_results[dialog_length], 2)}%')

    def index_generated_images(self):
        print('Preparing generated images...')

        if self.cfg['cache_gen_corpus'] and os.path.exists(self.cfg['cache_gen_corpus']):
            print(f"<<<< Cached generated corpus loaded: {self.cfg['cache_gen_corpus']} >>>>")
            print('Warning: make sure this cache matches the current image embedder.')
            self.generated_images = torch.load(self.cfg['cache_gen_corpus'])
            return

        generated_images = []
        generated_ids = []
        generated_dataloader = torch.utils.data.DataLoader(
            self.generated_images_dataset,
            batch_size=min(self.cfg['queries_bs'], len(self.generated_images_dataset)),
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

        for batch in tqdm.tqdm(generated_dataloader):
            batch_vectors = F.normalize(
                self.image_embedder.model(batch['image'].to(self.cfg['device'])), dim=-1
            )
            generated_images.append(batch_vectors)
            generated_ids.append(batch['idx'].to(self.cfg['device']))

        generated_images = torch.cat(generated_images)
        generated_ids = torch.cat(generated_ids)

        arg_ids = torch.argsort(generated_ids)
        generated_images = generated_images[arg_ids]
        generated_ids = generated_ids[arg_ids]

        self.generated_images = generated_ids, generated_images

        if self.cfg['cache_gen_corpus']:
            os.makedirs(os.path.dirname(self.cfg['cache_gen_corpus']), exist_ok=True)
            torch.save(self.generated_images, self.cfg['cache_gen_corpus'])

    def index_corpus(self):
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            print(f"<<<< Cached corpus loaded: {self.cfg['cache_corpus']} >>>>")
            print('Warning: make sure this cache matches the current image embedder.')
            self.corpus = torch.load(self.cfg['cache_corpus'])
            if self.cfg['use_generated_image']:
                self.index_generated_images()
            return

        corpus_dataloader = torch.utils.data.DataLoader(
            self.corpus_dataset,
            batch_size=min(self.cfg['queries_bs'], len(self.corpus_dataset)),
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

        print('Preparing corpus...')
        corpus_vectors = []
        corpus_ids = []

        for batch in tqdm.tqdm(corpus_dataloader):
            batch_vectors = F.normalize(
                self.image_embedder.model(batch['image'].to(self.cfg['device'])), dim=-1
            )
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(self.cfg['device']))

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)

        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]

        self.corpus = corpus_ids, corpus_vectors

        if self.cfg['cache_corpus']:
            os.makedirs(os.path.dirname(self.cfg['cache_corpus']), exist_ok=True)
            torch.save(self.corpus, self.cfg['cache_corpus'])

        if self.cfg['use_generated_image']:
            self.index_generated_images()


def get_first_hitting_time(target_recall, hitting_recall=10):
    """Return the first round at which each sample reaches Hits@k."""
    target_recalls = target_recall.view(11, -1).T
    hits = target_recalls < hitting_recall

    final_hits = torch.inf * torch.ones(target_recalls.shape[0])
    hitting_times = []
    for round_idx in range(11):
        round_hits = hits[:, round_idx]
        final_hits[round_hits] = torch.min(
            final_hits[round_hits],
            torch.ones(final_hits[round_hits].shape) * round_idx,
        )
        hitting_times.append(final_hits.clone())

    return torch.stack(hitting_times)


def cumulative_hits_per_round(target_recall, hitting_recall=10):
    """Return cumulative hit ratio up to each round."""
    if isinstance(hitting_recall, tuple):
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]
    ht_times = get_first_hitting_time(target_recall, hitting_recall)
    return (ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0]


if __name__ == '__main__':
    cfg = {
        'corpus_bs': 128,
        'queries_bs': 64,
        'num_workers': 0,
        'sep_token': ', ',
        'cache_corpus': '',
        'queries_path': 'dialogues/VisDial_v1_0_queries_val.json',
        'corpus_path': 'ChatIR_Protocol/Search_Space_val_50k.json',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'use_generated_image': False,
        'cache_gen_corpus': '',
    }

    with torch.no_grad():
        for query in ['VisDial_v1_0_queries_val.json']:
            cfg['queries_path'] = f'dialogues/{query}'
            print('Start evaluation on', cfg['queries_path'])

            print('Do not use generated images')
            cfg['use_generated_image'] = False
            cfg, dialog_encoder, image_embedder = build_model(cfg)
            evaluator = ChatIREval(cfg, dialog_encoder, image_embedder)
            evaluator.index_corpus()
            evaluator.run(hits_at=10)
            print('----------------------------------------------------')

            print('Use generated images with rerank fusion')
            cfg['use_generated_image'] = True
            cfg, dialog_encoder, image_embedder = build_model(cfg)
            evaluator = ChatIREval(cfg, dialog_encoder, image_embedder)
            evaluator.index_corpus()
            evaluator.run(hits_at=10)
            print('----------------------------------------------------')
