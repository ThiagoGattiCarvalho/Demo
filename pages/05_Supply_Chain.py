import ntplib
from time import ctime
import streamlit as st
import datetime
valid = datetime.datetime(2024, 6, 30, 12, 00, 00)
ntp_client = ntplib.NTPClient()

try:
    response = ntp_client.request('pool.ntp.org')                                                   # sometimes it may be out of air
    now = datetime.datetime.strptime(ctime(response.tx_time), "%a %b %d %H:%M:%S %Y")
    if now > valid:
        st.write('This module has expired. Please, contact Thiago to renew your plan.')
        st.stop()
except:
    pass

st.set_page_config(layout="wide")

st.title("Supply Chain")

st.markdown("You don't have access to this module. Please, contact Thiago.")

st.markdown("You have access to the Filter and the Analytics modules.")