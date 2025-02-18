import streamlit as st
import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

try:
    p=st.file_uploader("Upload a csv file")
 
    pf=pd.read_csv(p)
    df=pf[['ORDERDATE','SALES']]
    loop=0
    while(loop<2):
        outliers=[]
        def detect(df):
            threshold=3
            m=np.mean(df)
            deviation=np.std(df)
            
            for i in df:
                z_score=(i-m)/deviation
                if np.abs(z_score)>threshold:
                    outliers.append(i)
            return outliers
        outliers_pt=detect(df.SALES)
        outliers_pt=[*set(outliers_pt)]
        l=len(outliers_pt)
        o=[]
        n=0
        for i in df.SALES:
            for j in range(l):
                k=outliers_pt[j]
                if i == k:
                    o.append(n)
            n+=1
                    
        o=[*set(o)]


        for i in o:
            df=df.drop(i)
        loop += 1

    df['ORDERDATE']=pd.to_datetime(df.ORDERDATE, format='%d-%m-%Y')
    df.sort_values(by='ORDERDATE',inplace=True)
    df=df.groupby('ORDERDATE').sum()
    df=df.resample(rule='MS').sum()

    def error_check(test,test_pred):
        return (mean_absolute_percentage_error(test,test_pred)*100)
    

    sd=st.sidebar.radio("Navigation",['Home','Linear Regression','Random Forest','Moving Average','Holt Winter Additive','Holt Winter Multiplicative','ARIMA','SARIMA'])
    a = []
    a = [0 for i in range(7)]
    i=0

    if sd == 'Home':

        st.title("Home")
        st.text('The visual representation of the data is : ')
        st.dataframe(df)
        ts=df['SALES']
        plt.figure(figsize = (20,8))
        plt.plot(ts, label = 'SALES')
        plt.title('Time Series')
        plt.xlabel('Time(Year, Month)')
        plt.ylabel("SALES")
        plt.legend(loc = 'best')
        st.pyplot()


    if sd == 'Holt Winter Additive':

        st.title('Holt Winter Additive')
        st.text('The Holt winter Additive forecast of the data with 80 is to 20 rule : ')
        train=df[:26]
        test=df[26:]
        hwmodel=ExponentialSmoothing(train.SALES,trend='add',seasonal='add',seasonal_periods=12).fit()
        test_pred=hwmodel.forecast(7)
        train['SALES'].plot(legend=True,label='train',figsize=(20,8))
        test['SALES'].plot(legend=True,label='test',figsize=(20,8))
        test_pred.plot(legend=True,label='prediction')
        st.pyplot()
        if st.button('Calculate Error'):
            if a[0]==0:
                a[0]=error_check(test,test_pred)
                st.write("Mean Absolute Percentage Error is : ", round(a[0],2),'%')
            else:
                st.write("Mean Absolute Percentage Error is : ", round(a[0],2),'%')

        hwmodel0=ExponentialSmoothing(df.SALES,trend='add',seasonal='add',seasonal_periods=12).fit()
        def graph(k):
            pred=hwmodel0.forecast(k)
            df['SALES'].plot(legend=True,label='train',figsize=(20,8))
            pred.plot(legend=True,label='Prediction')
            st.pyplot()
        k = st.slider('Calculate for Future!!!',min_value=2,max_value=50,value = 12, step=2)
        graph(k)

    if sd == 'Holt Winter Multiplicative':

        st.title('Holt Winter Multiplicative')
        st.text('The Holt winter Multplicative forecast of the data with 80 is to 20 rule : ')
        train1=df[:26]
        test1=df[26:]
        hwmodel1=ExponentialSmoothing(train1.SALES,trend='mul',seasonal='mul',seasonal_periods=12).fit()
        test_pred1=hwmodel1.forecast(7)
        train1['SALES'].plot(legend=True,label='train',figsize=(20,8))
        test1['SALES'].plot(legend=True,label='test',figsize=(20,8))
        test_pred1.plot(legend=True,label='prediction')
        st.pyplot()
        if st.button('Calculate Error'):
            if a[1]==0:
                a[1]=error_check(test1,test_pred1)
                st.write("Mean Absolute Percentage Error is : ", round(a[1],2),'%')
            else:
                st.write("Mean Absolute Percentage Error is : ", round(a[1],2),'%')
        hwmodel2=ExponentialSmoothing(df.SALES,trend='add',seasonal='add',seasonal_periods=12).fit()

        def graph(k):
            pred=hwmodel2.forecast(k)
            df['SALES'].plot(legend=True,label='train',figsize=(20,8))
            pred.plot(legend=True,label='Prediction')
            st.pyplot()
        k = st.slider('Calculate for Future!!!',min_value=2,max_value=50,value = 12, step=2)
        graph(k)

    if sd == 'Moving Average':

        st.title('Moving Average')
        st.text('Data Visualisation with moving average : ')

        df['SALES'].plot(legend=True,label='SALES',figsize=(20,8))
        MOV = df['SALES'].rolling(window=3).mean()
        MOV.plot(legend=True,label='Moving Average',figsize=(20,8))
        st.pyplot()
        mov=MOV.iloc[3:]
        p=df.iloc[3:]

        if st.button('Calculate Error'):
            a[2]=error_check(mov,p)
            st.write("Mean Absolute Percentage Error is : ", round(a[2],2),'%')

    if sd == 'ARIMA':
        
        st.title('ARIMA')
        st.text('ARIMA forecast with 80 is to 20 rule: ')

        from statsmodels.tsa.arima.model import ARIMA
        train2=df.iloc[:-4]
        test2=df.iloc[-4:]
        model = ARIMA(train2['SALES'],order=(0,0,1))
        model = model.fit()
        start = len(train2)
        end = len(test2) + len(train2) - 1
        pred = model.predict(start=start,end=end,type = 'levels') #Mentioned from which part to which part the prediction should be done
        pred.index=df.index[start:end+1]
        
        pred.plot(legend=True,label='Arima Prediction')
        train2['SALES'].plot(legend=True,label='Train')
        test2['SALES'].plot(legend=True,label='Test',figsize=(20,8))
        st.pyplot()

        if st.button('Calculate Error'):
            temp=test2
            a[3]=error_check(temp,pred)
            st.write("Mean Absolute Percentage Error is : ", round(a[3],2),'%')

        def graph(k):
            model2 = ARIMA(df['SALES'],order=(0,0,1))
            model2 = model2.fit()
            pred=model2.predict(start=len(df),end=len(df)+k,type='levels').rename('ARIMA Predictions')
            df.plot()
            pred.plot(legend=True,label='ARIMA Prediction',figsize=(20,8))
            st.pyplot()
        k = st.slider('Calculate for Future!!!',min_value=2,max_value=50,value = 12, step=2)
        graph(k)

    if sd == 'SARIMA':

        st.title('SARIMA')
        st.text('SARIMA forecast with 80 is to 20 rule : ')

        import statsmodels.api as sm
        train3=df[:26]
        test3=df[26:]
        start = len(train3)
        end = len(test3) + len(train3) -1
        model3 = sm.tsa.statespace.SARIMAX(train3['SALES'],order=(4,0,4),seasonal_order=(4,0,4,12))
        results = model3.fit()
        pred2=results.predict(start=start,end=end,dynamic=True)

        pred2.plot(legend=True)
        train3['SALES'].plot(legend=True,label='Train')
        test3['SALES'].plot(legend=True,label='Test',figsize=(20,8))
        st.pyplot()

        if st.button('Calculate Error'):
            temp=test3
            a[4]=error_check(temp,pred2)
            st.write("Mean Absolute Percentage Error is : ", round(a[4],2),'%')

        def graph(k):
            model = sm.tsa.statespace.SARIMAX(df['SALES'],order=(4,0,4),seasonal_order=(4,0,4,12))
            results = model.fit()
            pred3=results.predict(start=len(df),end=len(df)+k,dynamic=True)
            df['SALES'].plot(legend=True,label='SALES')
            pred3.plot(legend=True,label='Prediction',figsize=(20,8))
            st.pyplot()
        k = st.slider('Calculate for Future!!!',min_value=2,max_value=50,value = 12, step=2)
        graph(k)
        
    if sd == 'Linear Regression':

        st.title('Linear Regression')
        st.text('Linear Regression Visualisation of data : ')

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        lr=LinearRegression()
        df['Sales_last_month']=df['SALES'].shift(+1)
        df['Sales_last_2months']=df['SALES'].shift(+2)
        df=df.dropna()
        x1,x2,y=df['Sales_last_month'],df['Sales_last_2months'],df['SALES']
        x1,x2,y=np.array(x1),np.array(x2),np.array(y)
        x1,x2,y=x1.reshape(-1,1),x2.reshape(-1,1),y.reshape(-1,1) #how many ever the rows under 1 column
        final_x=np.concatenate((x1,x2),axis=1)
        X_train,X_test,y_train,y_test=final_x[:-16],final_x[-16:],y[:-16],y[-16:]
        lr.fit(X_train,y_train)
        lr_pred=lr.predict(X_test)
        plt.rcParams['figure.figsize']=(16,4)
        plt.plot(lr_pred,label='LinearRegression_prediction')
        plt.plot(y_test,label='Actual test')
        plt.legend(loc='upper left')
        st.pyplot()
        
        if st.button('Calculate Error'):
            a[5]=error_check(y_test,lr_pred)
            st.write("Mean Absolute Percentage Error is : ", round(a[5],2),'%')

    if sd == 'Random Forest':

        st.title('Random Forest')
        st.text('Visualisation of data through Random Forest Regressor')

        from sklearn.ensemble import RandomForestRegressor
        model=RandomForestRegressor(n_estimators=100,max_features=3,random_state=1)
        df['Sales_last_month']=df['SALES'].shift(+1)
        df['Sales_last_2months']=df['SALES'].shift(+2)
        df=df.dropna()
        x1,x2,y=df['Sales_last_month'],df['Sales_last_2months'],df['SALES']
        x1,x2,y=np.array(x1),np.array(x2),np.array(y)
        x1,x2,y=x1.reshape(-1,1),x2.reshape(-1,1),y.reshape(-1,1) #how many ever the rows under 1 column
        final_x=np.concatenate((x1,x2),axis=1)
        X_train,X_test,y_train,y_test=final_x[:-16],final_x[-16:],y[:-16],y[-16:]
        model.fit(X_train,y_train)
        rf_pred=model.predict(X_test)
        plt.rcParams['figure.figsize']=(16,4)
        plt.plot(rf_pred,label='Random_forest_prediction')
        plt.plot(y_test,label='Actual test')
        plt.legend(loc='upper left')
        st.pyplot()

        if st.button('Calculate Error'):
            a[6]=error_check(y_test,rf_pred)
            st.write("Mean Absolute Percentage Error is : ", round(a[6],2),'%')

except ValueError:
    st.error('Please insert a valid CSV file is not inserted')

except :
    st.error('Invalid CSV file')