from dataset import dataset 


class CSVtoTS():
    """Convert raw csv data into format time series models assume"""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    
class CSVtoStaticGraph():
    """
    Convert raw csv data into format static graph 
    models assume (e.g. node list and edge list)
    """
    
    def __init__(self):
        self.dataset = dataset
    
class CSVtoDynamicGraph():
    """Convert raw csv data into format dynamic graph
    models assume (e.g. node list and edge list with timestamps or sequence index)
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def preprocessDataset(self, dataset):
        """pre-process dataset"""
        pass

    def generateDataset(dataset, snap_size, train_per=0.5, anomaly_per=0.01):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc'], default='uci')
    parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
    parser.add_argument('--train_per', type=float, default=0.5)
    args = parser.parse_args()

    snap_size_dict = {'uci':1000, 'digg':6000, 'btc_alpha':1000, 'btc_otc':2000}

    if args.anomaly_per is None:
        anomaly_pers = [0.01, 0.05, 0.10]
    else:
        anomaly_pers = [args.anomaly_per]

    for anomaly_per in anomaly_pers:
        generateDataset(args.dataset, snap_size_dict[args.dataset], train_per=args.train_per, anomaly_per=anomaly_per)