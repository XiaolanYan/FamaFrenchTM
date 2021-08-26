import datetime
import pandas as pd
import numpy as np
from MutualFundScore.common import *

from tqdm import tqdm
import statsmodels.api as sm

def regress(fundid, df, normalize = False,printResult = False):
    '''
    @fundid:"0000001.SZ"
    @df: pd.DataFrame(), columns = ["Rm","Rm2","HML","HML2","SMB","SMB2","return]
    @normalize: bool,
    @prinResult: bool,
    '''
    df.dropna(inplace =True)
    x = df[["Rm","Rm2","HML","HML2","SMB","SMB2"]]
    y = df[["return"]]

    #
    factorNames = list(x.columns)

    if normalize:
        x = (x - np.mean(x)) / np.std(x)

    #
    x = sm.add_constant(x)

    # GLS
    model = sm.GLS(y, x).fit()

    #
    if printResult:
        print(model.summary2())

    #

    params = model.params
    tvalues =model.tvalues
    tvalues.rename(lambda x: x + str("_t"), axis='columns',inplace =True)
    pvalues = model.pvalues
    pvalues.rename(lambda x: x + str("_p"), axis='columns',inplace =True)

    res = params.append(tvalues)
    res = res.append(pvalues)
    res["fundid"] = fundid
    res = pd.DataFrame(res)
    res = res.transpose()

    return res



def regression_from_calculated_factors(database):

    startdate = datetime.datetime(2016, 1, 1)
    enddate = datetime.datetime(2020, 11, 1)

    #
    # get_regression_data(database,startdate,enddate,"000002.SZ")
    market_return = database.Get_Daily_Bar("000001.SH", instrument_type="Index", datetime1=startdate, datetime2=enddate)
    market_return = pd.DataFrame(market_return)
    market_return["Rm"] = (market_return["close"] - market_return["close"].shift(periods=1)) / market_return["close"].shift(periods=1)
    market_return["Rm2"] = market_return["Rm"] * market_return["Rm"]
    market_return = market_return[["date", "Rm", "Rm2"]]
    market_return.set_index("date", inplace=True)

    SMB_HML = pd.read_csv("factors.csv")
    SMB_HML['date'] = SMB_HML['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"))
    SMB_HML.set_index("date", inplace=True)
    SMB_HML = SMB_HML[["SMB", "HML"]]
    SMB_HML["SMB2"] = SMB_HML["SMB"] * SMB_HML["SMB"]
    SMB_HML["HML2"] = SMB_HML["HML"] * SMB_HML["HML"]
    all_factors = market_return.join([SMB_HML], how="inner")
    # all_factors.to_csv("all_factors.csv")
    fundlist = database.Get_Instruments_DataFrame(instrument_type="mutualfund")  # ,filter={"invest_type1" :'股票型基金'})
    fundlist = fundlist[["symbol"]]
    fundlist["symbol"] = fundlist["symbol"].apply(lambda x: x.decode("utf-8"))
    funds_list = list(fundlist["symbol"])
    res = pd.DataFrame()
    for fundID in tqdm(funds_list):
        try:
            funds = database.Get_Daily_Bar(fundID, instrument_type="mutualfund", datetime1=startdate, datetime2=enddate)
            if len(funds) < 100:
                continue
            funds = pd.DataFrame(funds)
            funds["return"] = (funds["adjusted_net_asset_value"] - funds[
                "adjusted_net_asset_value"].shift(periods=1)) / funds["adjusted_net_asset_value"].shift(periods=1)
            funds = funds[["return", "date"]]
            funds.set_index("date", inplace=True)

            rdf = funds.join([SMB_HML, market_return], how="inner")

            subres = regress(fundID, rdf)
            res = res.append(subres)
        except KeyError:
            print("{} has no data in the given time period".format(fundID))
            continue
        except Exception:
            print("fund id is:{}, Unknown Exception".format(fundID))
            continue
    res.set_index("fundid", inplace=True)
    res.to_csv("fund_all.csv")

def regression_from_download_factors(database):

    startdate = datetime.datetime(2016, 1, 1)
    enddate = datetime.datetime(2020, 11, 1)


    SMB_HML = pd.read_csv("csmar_factor_free_shares.csv")
    SMB_HML['date'] = SMB_HML['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"))
    SMB_HML.set_index("date", inplace=True)
    SMB_HML["SMB2"] = SMB_HML["SMB"] * SMB_HML["SMB"]
    SMB_HML["HML2"] = SMB_HML["HML"] * SMB_HML["HML"]
    SMB_HML["Rm2"] = SMB_HML["Rm"]*SMB_HML["Rm"]
    # all_factors.to_csv("all_factors.csv")
    fundlist = database.Get_Instruments_DataFrame(instrument_type="mutualfund",filter={"invest_type1" :'股票型基金'})
    fundlist = fundlist[["symbol"]]
    fundlist["symbol"] = fundlist["symbol"].apply(lambda x: x.decode("utf-8"))
    funds_list = list(fundlist["symbol"])
    res = pd.DataFrame()
    for fundID in tqdm(funds_list):
        try:
            funds = database.Get_Daily_Bar(fundID, instrument_type="mutualfund", datetime1=startdate, datetime2=enddate)
            if len(funds) < 100:
                continue
            funds = pd.DataFrame(funds)
            funds["return"] = (funds["adjusted_net_asset_value"]-funds["adjusted_net_asset_value"].shift(periods=1) ) / funds["adjusted_net_asset_value"].shift(periods=1)
            funds = funds[["return", "date"]]
            funds.set_index("date", inplace=True)

            rdf = funds.join([SMB_HML], how="inner")

            subres = regress(fundID, rdf)
            res = res.append(subres)
        except KeyError:
            print("{} has no data in the given time period".format(fundID))
            continue
        except Exception:
            print("fund id is:{}, Unknown Exception".format(fundID))
            continue
    res.set_index("fundid", inplace=True)
    res.to_csv("stockfund_csmar.csv")


if __name__ == '__main__':
    #
    from Core.Config import *

    pathfilename = os.getcwd() + "\..\Config\config2.json"
    config = Config(pathfilename)
    database = config.DataBase("JDMySQL")
    regression_from_download_factors(database)
    startdate = datetime.datetime(2016,1,1)
    enddate = datetime.datetime(2020,12,1)

