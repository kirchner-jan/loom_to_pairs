from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob, jsonlines, pickle, gc, torch

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache() 
    # Load the model
    model = SentenceTransformer('all-mpnet-base-v2')
    max_seq_len = model.get_max_seq_length()
    # Find processed datasets
    file_list = glob.glob('data/*.jsonl')
    for file_name in file_list:
        print('Now processing: ', file_name)
        # Load the dataset
        embed_stubs = []
        num_lines = sum(1 for _ in open(file_name))
        with tqdm(jsonlines.open(file_name, mode='r'), total=num_lines, unit="json files") as reader:
            for obj in reader:
                # Embed the sentences
                embed_stubs += [ [(stub[0], stub[1], model.encode(stub[0][-200:])) for stub in obj] ]
        # Save the embeddings
        pickle.dump(embed_stubs, open(file_name[:-6] + '.pkl', 'wb'))