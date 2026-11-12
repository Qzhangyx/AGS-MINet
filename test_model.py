import shutil
import argparse
from functools import partial
import torch
import esm
from torch_geometric.transforms import Compose
import numpy as np
from models.PD import Pocket_Design_new, sample_from_categorical
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.transforms import *
from torch.utils.data import DataLoader
import wandb
import copy, pickle
from tqdm import tqdm
from vina_evalution import vina_mp, calculate_vina
import copy
from models.PD import RES_ATOM14, NUM_ATOMS, AA_NUMBER_NAME

from utils.protein_ligand import PDBProtein


def update_dataset_paths(dataset):
    new_data = []
    for i in range(len(dataset)):
        item = copy.deepcopy(dataset[i])  
        item['protein_filename'] = os.path.join(
            "../data1/crossdocked_v1.1_rmsd1.0", 
            item['protein_filename'])
        item['ligand_filename'] = os.path.join(
            "../data1/crossdocked_v1.1_rmsd1.0", 
            item['ligand_filename'])
        item['whole_protein_name'] = os.path.join(
            "../data1/crossdocked_v1.1_rmsd1.0", 
            item['whole_protein_name'])
        new_data.append(item)
    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model_cd.yml')
    parser.add_argument('--device', type=str, default='cuda:5')
    parser.add_argument('--logdir', type=str, default='./logs_cd')
    parser.add_argument('--target', type=str, default='./data1/crossdocked_v1.1_rmsd1.0/')
    
    args = parser.parse_args()
    config = load_config(args.config)
    seed_all(config.train.seed)

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,])

    # esm
    name = 'esm1b_t33_650M_UR50S'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()
    del pretrained_model

    print('Loading model ckp...')
    ckpt_path = "./logs_cd/cftfm_8_6_test_0/checkpoints/300000.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    print('Building model...')
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)  
      
    model.load_state_dict(ckpt['model'])
    
    # Datasets and loaders
    print('Loading test dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform)
    test_set = subsets['test']

    num_rounds = 3
    num_samples = 100
    results = {}

    parent_dir = os.path.dirname(os.path.dirname(ckpt_path))
    iter_num = os.path.basename(ckpt_path).replace(".pt", "")
    save_dir = os.path.join(parent_dir, "vina_data", iter_num)
    new_test_set = update_dataset_paths(test_set)
    

    for i in range(len(test_set)):
        try:
            
            print("process", i)
            protein_filename = new_test_set[i]['protein_filename']
            ligand_filename = new_test_set[i]['ligand_filename']
            dir_name = protein_filename[:-4]
            pocket_name = os.path.basename(dir_name)

            results[i] = {"rounds": []}

            for r in range(num_rounds):


                final_path = os.path.join(save_dir, f"round_{r}", pocket_name)
                if not os.path.exists(final_path):
                    os.makedirs(final_path, exist_ok=True)

                aar_list = []
                rmsd_list = []
                datalist = [new_test_set[i] for _ in range(num_samples)]
                test_loader = DataLoader(datalist, batch_size=10, shuffle=False,
                                        num_workers=config.train.num_workers,
                                        collate_fn=partial(collate_mols_block, batch_converter=batch_converter))

                model.generate_id = 0
                model.generate_id1 = 0

                with torch.no_grad():
                    model.eval()
                    for batch in tqdm(test_loader):
                        for key in batch:
                            if torch.is_tensor(batch[key]):
                                batch[key] = batch[key].to(args.device)

                        aar, rmsd, _ = model.generate(batch, final_path)
                        aar_list.append(aar.item())
                        rmsd_list.append(rmsd.item())


                aar_array = np.array(aar_list)
                rmsd_array = np.array(rmsd_list)

                result = {
                    "mean_aar": aar_array.mean(),
                    "mean_rmsd": rmsd_array.mean(),
                }

                results[i]["rounds"].append(result)
                torch.save(results,os.path.join(save_dir, "result_temp.pt"))

        except Exception as e:
            print(f"{i} error:", e)
    torch.save(results, os.path.join(save_dir, "result.pt"))

