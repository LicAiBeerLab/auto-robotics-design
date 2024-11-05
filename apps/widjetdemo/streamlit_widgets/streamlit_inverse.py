import streamlit as st
import numpy as np

st.session_state.stage = 'class_choice'

if st.session_state.stage == 'class_choice':
    col_1, col_2, col_3 = st.columns(3, gap="medium")
    with col_1:
        st.button(label='free ', key='return_to_topology_choice', on_click=lambda: st.session_state.__setitem__('stage', 'topology_choice'))