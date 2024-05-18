import streamlit as st
html_temp = """
<div style="background-color:orange;padding:10px">
<h2 style="color:white;text-align:center;">WELCOME</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write("----")
st.title(':robot_face:')
st.title(":orange[Pdf Analysis Application with Gemini 1.5 pro]" )

st.subheader("Please use the sidebar menu to select different functions to analyse your pdf.")  