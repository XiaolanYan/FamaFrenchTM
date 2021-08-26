import datetime
import pandas as pd
import numpy as np
from MutualFundScore.common import *
import gc
from tqdm import tqdm
import statsmodels.api as sm



def calc_group_return(df,targetname, isweighted):
    if isweighted:
        group = df.groupby(['date']).agg({'CAP_return': np.sum, "CAP": np.sum})
        group["weighted_return"] = group["CAP_return"] / group["CAP"]
        group.rename(columns={"weighted_return": targetname}, inplace=True)
    else:
        group = df.groupby(['date']).agg({'return': np.mean})
        group.rename(columns={"return": targetname}, inplace=True)
    return group[[targetname]]

def get_2x3_portfolio(CAPData,PBData):
    '''
        :
        :return:[{"date":"2020-01-10","SV":[],"SN":[],"SG":[],"BV":[],"BN":[],"BG":[]}]
    '''

    holdings =[]

    PBdatelist = PBData["date"].drop_duplicates()
    PBdatelist =sorted(PBdatelist)
    CAPdatelist = CAPData["date"].drop_duplicates()
    CAPdatelist = sorted(CAPdatelist)

    startdate = max(PBdatelist[0],CAPdatelist[0])

    datelist = list(set(PBdatelist+CAPdatelist))
    datelist =list(sorted(datelist))

    startpoint = datelist.index(startdate)
    datelist =datelist[startpoint:]

    isCAPstart = 0
    isPBstart =0
    for date in datelist:
        CAPsub = CAPData[CAPData['date'] == date]
        CAPsub.dropna(axis='index', how='all', subset=["value"], inplace=True)
        CAPsub =CAPsub.sort_values(by='value')
        if len(CAPsub) > 100:
            isCAPstart=1
            subNum =int(len(CAPsub)*0.5)
            Big = CAPsub.tail(subNum)
            Small =CAPsub.head(subNum)
            Big = list(Big["symbol"])
            Small =list(Small["symbol"])

        PBsub = PBData[PBData['date'] == date]
        PBsub.dropna(axis='index', how='all', subset=["value"], inplace=True)
        PBsub = PBsub.sort_values(by='value')
        if len(PBsub) > 100:
            isPBstart = 1
            subNum = int(len(PBsub) * 0.3)
            # 数据库里面是price to book,SMB是book to price来排序
            Value = PBsub.head(subNum)
            Growth = PBsub.tail(subNum)
            p1 = float(Value.tail(1)['value'])
            p2 = float(Growth.head(1)['value'])
            Neutral = PBsub.loc[PBsub["value"]<p2][PBsub["value"]>p1]
            Growth = list(Growth["symbol"])
            Value = list(Value["symbol"])
            Neutral = list(Neutral["symbol"])
        if isCAPstart*isPBstart ==0:
            continue
        SV =list(set(Small).intersection(set(Value)))
        SN =list(set(Small).intersection(set(Neutral)))
        SG =list(set(Small).intersection(set(Growth)))

        BV =list(set(Big).intersection(set(Value)))
        BN =list(set(Big).intersection(set(Neutral)))
        BG =list(set(Big).intersection(set(Growth)))

        res = {"date":date,"SV":SV,"SN":SN,"SG":SG,"BV":BV,"BN":BN,"BG":BG}

        holdings.append(res)
    return holdings

def get_SMB_portfolio(CAPData):
    '''
    :
    :return:[{"date":"2020-01-10","S_portfolio":["000001.SZ","000002.SZ"],"B_portfolio":["000005.SH","3000015.SH"]}]

    '''

    #if datetime does not exists: return null
    #else: calc
    # CAPData = database.GetDataFrame("factor", "lncap")
    # CAPData =CAPData[CAPData["date"]>=startdate & CAPData["date"]<=enddate]
    datelist = CAPData["date"].drop_duplicates()
    SMB_porfolio =[]
    for date in datelist:
        CAPsub = CAPData[CAPData['date'] == date]
        CAPsub.dropna(axis='index', how='all', subset=["value"], inplace=True)
        CAPsub =CAPsub.sort_values(by='value')
        Num = len(CAPsub)
        if Num < 100: continue
        subNum =int(Num*0.5)
        Big_size = CAPsub.tail(subNum)
        Small_size =CAPsub.head(subNum)
        res = {"date":date,"B_portfolio":list(Big_size["symbol"]),"S_portfolio":list(Small_size["symbol"])}
        SMB_porfolio.append(res)


    return SMB_porfolio

def get_HML_portfolio(PBData):
    '''
    :
    :return:[{"date":"2020-01-10","H_portfolio":["000001.SZ","000002.SZ"],"L_portfolio":["000005.SH","3000015.SH"]}]

    '''

    # if datetime does not exists: return null
    # else: calc
    # PBData = database.GetDataFrame("factor", "pb_lf")
    # PBData = PBData[PBData["date"] >= startdate & PBData["date"] <= enddate]
    datelist = PBData["date"].drop_duplicates()
    datelist =sorted(datelist)
    HML_porfolio = []
    for date in datelist:
        PBsub = PBData[PBData['date'] == date]
        PBsub = PBsub.sort_values(by='value')
        PBsub = PBsub[PBsub['value']>0]
        Num = len(PBsub)
        subNum = int(Num*0.3)
        if Num<100:continue
        Growth = PBsub.tail(subNum)
        Value = PBsub.head(subNum)
        res = {"date": date, "H_portfolio": list(Value["symbol"]), "L_portfolio": list(Growth["symbol"])}
        HML_porfolio.append(res)

    return HML_porfolio


def calc_facrors_simple_divided(df,SMB_pf,HML_pf,isweighted):
    '''
    df: "date","symbol","return","CAP"
    SMB_df:
    HML_df:

    return:{"date":[],"SMB":[],"HML":[]}

    CAP could be total shares or free shares, depends on user

    '''

    #

    df["CAP_return"] = df["CAP"]*df["return"]
    SMB_df = pd.DataFrame()
    for i,line in enumerate(SMB_pf):
        if i ==len(SMB_pf)-1:
            continue

        date1 = SMB_pf[i]['date']
        date1 = datetime.datetime.strptime(str(date1), "%Y-%m-%d %H:%M:%S")


        date2 = SMB_pf[i+1]['date']
        date2 = datetime.datetime.strptime(str(date2), "%Y-%m-%d %H:%M:%S")

        sub_df = df[df['date']>=date1][ df['date']<date2]
        Small = line["S_portfolio"]
        # Small = [str(i) for i in Small]
        Big = line["B_portfolio"]
        # Big =[str(i) for i in Big]
        Small_df = sub_df[sub_df["symbol"].isin(Small)]
        Big_df = sub_df[sub_df["symbol"].isin(Big)]
        #calc SMB

        group_small = calc_group_return(Small_df,"small",isweighted)

        group_big = calc_group_return(Big_df,"big",isweighted)
        group_market = calc_group_return(sub_df,"Rm",isweighted)
        SMB = group_small.join([group_big,group_market], how="outer")
        SMB["SMB"] = SMB["small"] - SMB["big"]
        SMB_df = SMB_df.append(SMB)
        del Small_df,Big_df,sub_df,group_small, group_big, SMB
        gc.collect()
    # SMB_df.to_csv("SMB.csv")


    HML_df = pd.DataFrame()
    for i, line in enumerate(HML_pf):
        if i == len(HML_pf) - 1:
            continue
        date1 = HML_pf[i]['date']
        date1 = datetime.datetime.strptime(str(date1), "%Y-%m-%d %H:%M:%S")

        date2 = HML_pf[i + 1]['date']
        date2 = datetime.datetime.strptime(str(date2), "%Y-%m-%d %H:%M:%S")

        sub_df = df[df['date'] >= date1][df['date'] < date2]
        High = line["H_portfolio"]
        High =[str(i) for i in High]
        Low = line["L_portfolio"]
        Low =[str(i) for i in Low]
        High_df = sub_df[sub_df['symbol'].isin(High)]
        Low_df = sub_df[sub_df["symbol"].isin(Low)]
        # calc SMB

        group_high = calc_group_return(High_df,"high",isweighted)
        group_low = calc_group_return(Low_df,"low",isweighted)

        HML = group_high.join([group_low], how="outer")
        HML["HML"] = HML["high"] - HML["low"]
        HML_df = HML_df.append(HML)
        del High_df,Low_df,sub_df,group_high, group_low, HML
        gc.collect()
    factors = HML_df.join([SMB_df], how="outer")
    factors.to_csv("factors.csv")
    return factors

def calc_factors_2x3_divided(holding,stockdf):

    # res = {"date": date, "SV": SV, "SN": SN, "SG": SG, "BV": BV, "BN": BN, "BG": BG}
    stockdf["CAP_return"] = stockdf["CAP"]*stockdf["return"]
    factors =pd.DataFrame()
    for i,line in enumerate(holding):
        if i ==len(holding)-1:
            continue

        date1 = holding[i]['date']
        date1 = datetime.datetime.strptime(str(date1), "%Y-%m-%d %H:%M:%S")


        date2 = holding[i+1]['date']
        date2 = datetime.datetime.strptime(str(date2), "%Y-%m-%d %H:%M:%S")

        sub_df = stockdf[stockdf['date']>=date1][stockdf['date']<date2]

        SV,SN,SG,BV,BN,BG = line['SV'],line['SN'],line['SG'],line['BV'],line['BN'],line['BG']

        SV_df = sub_df[sub_df["symbol"].isin(SV)]
        SN_df = sub_df[sub_df["symbol"].isin(SN)]
        SG_df = sub_df[sub_df["symbol"].isin(SG)]
        BV_df = sub_df[sub_df["symbol"].isin(BV)]
        BN_df = sub_df[sub_df["symbol"].isin(BN)]
        BG_df = sub_df[sub_df["symbol"].isin(BG)]

        SVgroup = calc_group_return(SV_df,"SV",True)
        SNgroup = calc_group_return(SN_df,"SN",True)
        SGgroup = calc_group_return(SG_df,"SG",True)
        BVgroup = calc_group_return(BV_df,"BV",True)
        BNgroup = calc_group_return(BN_df,"BN",True)
        BGgroup = calc_group_return(BG_df,"BG",True)

        Marketgroup = calc_group_return(sub_df,"Rm",True)

        groups = SVgroup.join([SNgroup,SGgroup,BVgroup,BNgroup,BGgroup,Marketgroup],how="outer" )
        groups["SMB"] = (groups["SV"]+groups["SN"]+groups["SG"] -(groups["BV"]+groups["BN"]+groups["BG"]))/3
        groups["HML"] = ((groups["SV"]+groups["BV"])-(groups["SG"]+groups["BG"]))/2
        factors = factors.append(groups)
    return factors



def get_stock_data(database,startdate,enddate):
    '''
    this function calculates stock return of all the stocks listed in the database and saves data to local address
    '''

    all_stocks = database.Get_Instruments_DataFrame("stock")
    # all_stocks = database.GetDataFrame("financial_data", "instrument_stock", datetime1=startdate, datetime2=enddate)
    symbols = all_stocks["symbol"]
    all_data = pd.DataFrame()
    for symbol in tqdm(symbols):
        try:
            symbol_str = symbol.decode('utf-8')
            data = database.Get_Daily_Bar_DataFrame(symbol_str, "stock", datetime1=startdate, datetime2=enddate)
            if len(data)<1:
                print(symbol_str)
                continue
            data["return"] = (data["close"]-data["close"].shift(periods=1)) / data["close"].shift(periods=1)
            data = data[["date", "symbol", "return","total_shares","free_float_shares"]]
            all_data = all_data.append(data)
        except KeyError:
            print(symbol_str)
            continue
    all_data["symbol"] = all_data["symbol"].apply(lambda x: x.decode('utf-8'))
    str_startdate = datetime.date.strftime(startdate,"%Y%m%d")
    str_enddate = datetime.date.strftime(enddate,"%Y%m%d")
    file_name = "stock_data{}to{}.csv".format(str_startdate,str_enddate)
    all_data.to_csv(file_name)
    return all_data









def get_factors(stocks_filename,PBdata_filename,CAPdata_filename, startdate,enddate, isweighted = False, isSimpleDivided = True,weightedBy = "free_float_shares"):
    '''
    this function calculates factors using self-write functions

    isweighted:
                True: in the sub group, the return of the group is calculated weighted by capital
                False: in the sub group, the return is equally weighted

    isSimpleDivided:
                True: divide Cap group to 2 groups and PB to 3 groups,  SMB = small - big, HML = value -growth
                False: refer to the 2x3 divide method in Fama-French(1993)
                      SMB = 1/3(small Value+Small Neutral+small growth)- 1/3(big value+big neutral+big growth)
                      HML = 1/2(small value +big value)-1/2(small growth+big growth)
                      small value means the intersection of small group and value group

    weightedBy: "free_float_shares"
                "total_shares"
    '''


    # '''
    str_startdate = datetime.date.strftime(startdate, "%Y%m%d")
    str_enddate = datetime.date.strftime(enddate, "%Y%m%d")

    # PBData = database.GetDataFrame("factor", "pb_lf", datetime1=startdate, datetime2=enddate)
    # CAPData = database.GetDataFrame("factor", "lncap", datetime1=startdate, datetime2=enddate)
    PBData = pd.read_csv(PBdata_filename)
    # PBData['symbol'] = PBData['symbol'].apply(lambda x:x.decode('utf-8'))
    PBData = PBData[['symbol', 'date', 'value']]
    PBData['date'] = PBData['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    CAPData = pd.read_csv(CAPdata_filename)
    # CAPData['symbol'] = CAPData['symbol'].apply(lambda x:x.decode('utf-8'))
    CAPData = CAPData[['symbol','date','value']]
    CAPData['date'] = CAPData['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    # CAPData.to_csv(file_name_CAP)
    if isSimpleDivided:
        #step1: get portfolios of each group

        HML_pf = get_HML_portfolio(PBData)
        del PBData
        gc.collect()


        SMB_pf = get_SMB_portfolio(CAPData)
        del CAPData
        gc.collect()


        #step 2: calculate returns of the sub groups
        stocks = pd.read_csv(stocks_filename)
        stocks['date'] = stocks['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
        stocks["CAP"] = stocks[weightedBy]

        factors = calc_facrors_simple_divided(stocks,SMB_pf,HML_pf,isweighted)
        del stocks
        gc.collect()

    else:
        #step1: get portfolios of each group

        holdings = get_2x3_portfolio(CAPData,PBData)

        del CAPData,PBData
        gc.collect()

        #step 2: calculate returns of the sub groups
        str_startdate = datetime.date.strftime(startdate, "%Y%m%d")
        str_enddate = datetime.date.strftime(enddate, "%Y%m%d")
        file_name = "stock_data{}to{}.csv".format(str_startdate, str_enddate)
        stocks = pd.read_csv(file_name)
        stocks['date'] = stocks['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
        stocks["CAP"] = stocks[weightedBy]
        factors = calc_factors_2x3_divided(holdings,stocks)
        del stocks
        gc.collect()

    # factors.set_index("date",inplace=True)
    factors.to_csv("allfactors.csv")

    return factors



def get_factors_run(startdate,enddate,stocks_filename,PBdata_filename,CAPdata_filename):
    get_factors(stocks_filename,PBdata_filename,CAPdata_filename,startdate,enddate,isweighted=True,isSimpleDivided=True)




if __name__ == '__main__':
    #

    startdate = datetime.datetime(2010,1,1)
    enddate = datetime.datetime(2020,12,1)
    #step1: calcualte stock returns and save to local address
    # get_stock_data(database,startdate,enddate)
    #step2: calculate three factors,CAP,PB data read from database; stock return data read from local address, calculated in step 1
    stocks_filename = "stock_data20100101to20201201.csv"
    PBdata_filename = "PB_data20100101to20201201.csv"
    CAPdata_filename = "CAP_data20100101to20201201.csv"
    get_factors_run(startdate,enddate,stocks_filename,PBdata_filename,CAPdata_filename)
