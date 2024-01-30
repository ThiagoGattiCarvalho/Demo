import ntplib
from time import ctime
import streamlit as st
import datetime
valid = datetime.datetime(2024, 1, 30, 12, 00, 00)
ntp_client = ntplib.NTPClient()

try:
    response = ntp_client.request('pool.ntp.org')               # sometimes it may be out of air
    now = datetime.datetime.strptime(ctime(response.tx_time), "%a %b %d %H:%M:%S %Y")
    if now > valid:
        st.write('This module has expired. Please, contact Thiago to renew your plan.')
        st.stop()
except:
    pass

import sqlite3
import pandas as pd
import numpy as np

from sys import getsizeof

pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', -1)
st.set_page_config(layout="wide")

conn = sqlite3.connect('orkideon.sqlite')
df_model = pd.read_sql('select * from df_model', conn)
df_model_columns = list(pd.read_sql('select * from df_model_columns', conn).stack().values)
df_model_columns_num = list(pd.read_sql('select * from df_model_columns_num', conn).stack().values)
df_model_columns_int = list(pd.read_sql('select * from df_model_columns_int', conn).stack().values)
df_model_columns_float = list(pd.read_sql('select * from df_model_columns_float', conn).stack().values)
df_model_columns_cat = list(pd.read_sql('select * from df_model_columns_cat', conn).stack().values)
conn.close()

df_model_columns.sort()
df_model_columns_num.sort()
df_model_columns_cat.sort()

df_model[df_model_columns_num] = df_model[df_model_columns_num].apply(pd.to_numeric, errors='coerce')

try:
    df_model = df_model.sort_values(by=['Group', 'Period'], ascending=[True, True])
except:
    df_model = df_model.sort_values(by='Group', ascending=True)

try:                                                            # because re-imported loses Period
    df_model['Period'] = pd.to_datetime(df_model['Period'])
except:
    pass

df_model = df_model.fillna(value=np.nan)                        # to replace None by nan

df_model_columns_cat_all = ["All"] + df_model_columns_cat

conn = sqlite3.connect('orkideon.sqlite')
df_filtered = pd.read_sql('select * from df_filtered', conn)
df_filtered_columns = list(pd.read_sql('select * from df_filtered_columns', conn).stack().values)
df_filtered_columns_num = list(pd.read_sql('select * from df_filtered_columns_num', conn).stack().values)
df_filtered_columns_int = list(pd.read_sql('select * from df_filtered_columns_int', conn).stack().values)
df_filtered_columns_float = list(pd.read_sql('select * from df_filtered_columns_float', conn).stack().values)
df_filtered_columns_cat = list(pd.read_sql('select * from df_filtered_columns_cat', conn).stack().values)
conn.close()
del conn

df_filtered_columns.sort()
df_filtered_columns_num.sort()
df_filtered_columns_cat.sort()

df_filtered[df_filtered_columns_num] = df_filtered[df_filtered_columns_num].apply(pd.to_numeric, errors='coerce')

try:
    df_filtered = df_filtered.sort_values(by=['Group', 'Period'], ascending=[True, True])
except:
    df_filtered = df_filtered.sort_values(by='Group', ascending=True)

try:
    df_filtered['Period'] = pd.to_datetime(df_filtered['Period'])
except:
    pass

df_filtered = df_filtered.fillna(value=np.nan)

df_filtered_columns_cat_all = ["All"] + df_filtered_columns_cat


st.title("Filter")

with st.sidebar:
    st.markdown("You need to save the filter in order to see the changes in the table below.")

filter_filter, filter_admin = st.tabs(['Filter', 'Admin'])

with filter_filter:

    filter_filter1, filter_filter2, filter_filter3, filter_filter4, filter_filter5 = st.columns(5, gap='large')

    if df_model_columns_cat == []:
        df3 = df_model
    else:
        df3 = pd.DataFrame()

    with filter_filter1:
        
        if 'Group' not in df_model_columns:
            selected_columns = df_model_columns
        else:
            selected_columns = list(set(df_model_columns) - set(['Group']))

        selected_df_columns = st.multiselect(f"Columns to keep:", selected_columns, default=selected_columns)

        if 'Group' not in df_model_columns:
            pass
        else:
            selected_df_columns = selected_df_columns + ['Group']

        selected_df_model = df_model[selected_df_columns]
        selected_df_model_columns_num = list(selected_df_model.select_dtypes(include='number').columns)
        selected_df_model_columns_int = list(selected_df_model.select_dtypes(include=['int32','int64']).columns)
        selected_df_model_columns_float = list(selected_df_model.select_dtypes(include=['float32','float64']).columns)
        selected_df_model_columns_cat = list(selected_df_model.select_dtypes(include=['category', 'object']).columns)


    with filter_filter2:
        
        selected_df_model_columns_cat_ = list(set(selected_df_model_columns_cat) - set(['Group']))
        
        for column in selected_df_model_columns_cat_:

            if column == selected_df_model_columns_cat_[0]:
                options = selected_df_model[column].unique()
                try:
                    options.sort()
                except:
                    pass
                selected = st.multiselect(f"{column} filter:", options, default=options)
                df1 = selected_df_model.loc[selected_df_model[column].isin(selected)]
                df3 = pd.concat([df3, df1])

            else:
                options = df3[column].unique()
                try:
                    options.sort()
                except:
                    pass
                selected = st.multiselect(f"{column} filter:", options, default=options)
                df1 = df3.loc[df3[column].isin(selected)]
                df3 = df1


    with filter_filter3:

        for column in selected_df_model_columns_int:
            min_value = df3[column].min(skipna=True)
            max_value = df3[column].max(skipna=True)

            if max_value >= 1.797e+308:
                max_value = 1.797e+308

            if min_value <= -1.797e+308:
                min_value = -1.797e+308

            if min_value != max_value:
                selected_range = st.slider(f'{column} range:', min_value, max_value, (min_value, max_value))
                # df3 = df3[(df3[column] >= selected_range[0]) & (df3[column] <= selected_range[1])]                              # removes nan
                df3 = df3[(df3[column].isna() | ((df3[column] >= selected_range[0]) & (df3[column] <= selected_range[1])))]
            else:
                pass

    with filter_filter4:
        for column in selected_df_model_columns_float:
            min_value = df3[column].min(skipna=True)
            max_value = df3[column].max(skipna=True)

            if max_value >= 1.797e+308:
                max_value = 1.797e+308

            if min_value <= -1.797e+308:
                min_value = -1.797e+308

            if min_value != max_value:
                selected_range = st.slider(f'{column} range:', min_value, max_value, (min_value, max_value))
                df3 = df3[(df3[column].isna() | ((df3[column] >= selected_range[0]) & (df3[column] <= selected_range[1])))]
            else:
                pass


    with filter_filter5:
        if st.button('Save Filtered Data'):

            df_filtered_columns = pd.DataFrame(df3.columns)
            df_filtered_columns_num = pd.DataFrame(df3.select_dtypes(include='number').columns)
            df_filtered_columns_int = pd.DataFrame(df3.select_dtypes(include='int64').columns)
            df_filtered_columns_float = pd.DataFrame(df3.select_dtypes(include='float64').columns)
            df_filtered_columns_cat = pd.DataFrame(df3.select_dtypes(include=['category', 'object']).columns)

            conn = sqlite3.connect('orkideon.sqlite')

            df3.to_sql('df_filtered', conn, if_exists='replace', index=False)
            
            df_filtered_columns.to_sql('df_filtered_columns', conn, if_exists='replace', index=False)
            df_filtered_columns_num.to_sql('df_filtered_columns_num', conn, if_exists='replace', index=False)
            df_filtered_columns_int.to_sql('df_filtered_columns_int', conn, if_exists='replace', index=False)
            df_filtered_columns_float.to_sql('df_filtered_columns_float', conn, if_exists='replace', index=False)
            df_filtered_columns_cat.to_sql('df_filtered_columns_cat', conn, if_exists='replace', index=False)

            conn.close()
            del conn, df1

            st.rerun()

        if st.button('Export to .csv as Filtered Data Model'):
            try:
                conn = sqlite3.connect('orkideon.sqlite')
                df3.to_csv("df_filtered.csv", index=False, mode="w", encoding='iso-8859-1')
                conn.close()
                del conn
                st.text('Done!')
            except:
                st.text('Please, consolidate first.')
                st.stop()


    try:
        st.write(f"""##### Filtered Data Model {df_filtered.shape} (first 1000 rows)""")
        st.dataframe(df_filtered.head(1000))
        del df3
        del selected_columns, selected_df_columns, selected_df_model, selected_df_model_columns_num, selected_df_model_columns_int, selected_df_model_columns_float, selected_df_model_columns_cat
    except:
        pass




with filter_admin:

    del df_model, df_model_columns

    st.markdown("""##### In-Scope Variables""")
    in_scope_variables = dir()
    st.markdown(f"Active variables: {in_scope_variables}")
    st.markdown(f"Memory usage: {getsizeof(in_scope_variables)/1000} kb")
