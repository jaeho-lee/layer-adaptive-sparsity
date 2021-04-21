import torch,argparse,random,os
import numpy as np
from tools import *

""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
#parser.add_argument('--cuda', type=int, help='cuda number')
parser.add_argument('--model', type=str, help='network')
parser.add_argument('--pruner', type=str, help='pruning method')
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=20, help='start iteration for pruning')

args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model,amount_per_it,batch_size,opt_pre,opt_post = model_and_opt_loader(args.model,DEVICE)
train_loader, test_loader = dataset_loader(args.model,batch_size=batch_size)
pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader()

""" SET SAVE PATHS """
DICT_PATH = f'./dicts/{args.model}/{args.seed}'
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)
BASE_PATH = f'./results/iterate/{args.model}/{args.seed}'
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

""" PRETRAIN (IF NEEDED) """
if args.iter_start == 1:
    filename_string = 'unpruned.pth'
else:
    filename_string = args.pruner+str(args.iter_start-1)+'.pth'
if os.path.exists(os.path.join(DICT_PATH,filename_string)):
    print(f"LOADING PRE-TRAINED MODEL: SEED: {args.seed}, MODEL: {args.model}, ITER: {args.iter_start - 1}")
    state_dict = torch.load(os.path.join(DICT_PATH,filename_string),map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
else:
    if args.iter_start == 1:
        print(f"PRE-TRAINING A MODEL: SEED: {args.seed}, MODEL: {args.model}")
        pretrain_results = trainer(model,opt_pre,train_loader,test_loader)
        torch.save(pretrain_results, DICT_PATH+'/unpruned_loss.dtx')
        torch.save(model.state_dict(),os.path.join(DICT_PATH,'unpruned.pth'))
    else:
        raise ValueError('No (iteratively pruned/trained) model found!')

""" PRUNE AND RETRAIN """
results_to_save = []
for it in range(args.iter_start,args.iter_end+1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(model,amount_per_it)    
    result_log = trainer(model,opt_post,train_loader,test_loader)
    result_log.append(get_model_sparsity(model))
    results_to_save.append(result_log)
    torch.save(torch.FloatTensor(results_to_save),BASE_PATH+f'/{args.pruner}.tsr')
