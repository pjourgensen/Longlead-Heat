import sstpreparer as sst
import myconvlstm as clstm

if __name__ == '__main__':
    processor = sst.SSTPreparer('../Configs/sstpreparer_config.json',
                            download_raw_sst = False,
                            combine_raw_sst = False,
                            load_raw_t95 = False)
    constructor = clstm.MyConvLSTM('../Configs/myconvlstm_config.json',
                             Processor = processor)
    constructor.fit()
