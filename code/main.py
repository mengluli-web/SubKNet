import os 
from parameters import args_parser
from utils import *
from cvs import cv_process
from indep import indep_process
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    args = args_parser()
    if not os.path.exists('../data/split_data_'+args.dataset+'.npz'):
        split_datas(args.dataset,args.kfs,seed,args.types)
    if args.exp_name=='cv':
        cv_process()
    elif args.exp_name=='indep':
        indep_process()
        
        
        
        
        
        
        
        