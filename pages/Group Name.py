import streamlit as st
from PIL import Image

st.write("""    # ANGGOTA PROJECT AI4Y""")
st.write("# ")


if 'group.png' is not None:
    img = Image.open('group.png')
    img1 = img.resize((600, 600))
    st.image(img1, use_column_width=False)