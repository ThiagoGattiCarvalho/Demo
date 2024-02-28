# https://medium.com/@jcharistech/how-to-run-streamlit-apps-from-colab-29b969a1bdfc
#https://discuss.streamlit.io/t/how-to-launch-streamlit-app-from-google-colab-notebook/42399

import streamlit as st

# import pyodide_http
# pyodide_http.patch_all()        # this must be before importing requests
# import requests
from PIL import Image

st.set_page_config(layout="wide")

image_thiago = Image.open("Thiago.png") 
st.image(image_thiago, width = 250)
st.title("Thiago Gatti")
st.subheader("Supply Chain Planning & Machine Learning")

home1, home2 = st.columns(2, gap='large')

with home1:
    st.subheader("About Me")
    st.markdown("""
    \nDedicated and results-driven Supply Chain professional with easy transit between areas and a proven track record of driving business improvements.
    \nRecognized for integrating Finance and Supply Chain models to establish seamless Integrated Business Planning (IBP) and Autonomous Planning & Decision systems for substantial ROI improvement across various industries.
    \nAdept at combining strategic vision with hands-on problem-solving to drive innovation and operational excellence, leveraging a unique blend of technical proficiency and managerial acumen to elevate supply chain planning on forward-looking organizations.
    \nMA in Games Design in Germany, APICS SCOR-P in Belgium, APICS CSCP, MBA, Mechanical Engineering Bachelor, Python, Excel, SAP, 
    fluent English and Portuguese, average Italian, basic German.
    EU family and German driver's license.
    \nhttps://www.linkedin.com/in/thiagocscp/
    \nhttps://www.orkideon.com
    """)

    st.subheader("Hard Skills")
    st.markdown("""
    \n**Supply Chain Management:** Integrated Business Planning (IBP), Production Planning and Control (PPC), Demand Planning, Inventory Planning, Network Design;
    \n**Data Science and Machine Learning:** Statistical Modeling, Machine Learning Applications, Financial Modeling, Big Data Analysis;
    \n**Project Management:** Software Implementations, Custom Planning Tool Development, Process Redesign;
    \n**Consulting and Technical Proficiency:** Supply Chain Diagnosis, Solution Proposals, Process Improvement, Python (Pandas, Streamlit, Scikit-learn, Seaborn, Sqlite3, 
    Statsmodels, Scipy), Microsoft Excel & PowerPoint, JDA/Manugistics, Network Software, SCOR Framework, Algorithms, What-if Scenarios Simulation;
    """)

    st.subheader("Soft Skills")
    st.markdown("""
    \n**Leadership and Communication:** Team Building, Mentoring, Cross-Matrix Navigation, Effective Communication of Complex Concepts;
    \n**Adaptability and Continuous Learning:** Continuous Learning, Technological Adaptability, Motivation to Study;
    \n**Problem-Solving and Results-Oriented:** Diagnosis of Challenges, Innovative Solutions
    Delivering Results Regardless of Tools, Value-Added Approach, Well-Organized;
    \n**Interpersonal Skills and Strategic Thinking:** Diplomacy, Collaboration, Curriculum Development (Teaching and Research), Strategic Supply Chain Planning, Goal Setting;
    """)

    st.subheader("Successes")
    st.markdown("""
    \n**Founder & Supply Chain Data Scientist (May 2021 - Present):**
    \n•	Leveraged machine learning for end-to-end supply chain planning, integrating financial and statistical modeling.
    \n•	Achieved notable results, including over 10% ROI with proprietary IBP/S&OP approach and 99% forecast accuracy in demand planning.

    \n**Edgewell Personal Care - Supply Chain Business Process Improvement Manager for Europe (Sep 2018 - Dec 2020):**
    \n•	Diagnosed ROI improvement opportunities, reducing stocks, implementing Economic Order Quantities, and developing Open-To-Buy Production Planning Model.
    \n•	Led a transformative project in the Czech Republic and designed/implemented an Advanced S&OP/IBP tool, providing significant inventory insights.

    \n**Cologne University of Applied Sciences - Data Scientist and Researcher (Apr 2017 - Dec 2017):**
    \n•	Conducted statistical analyses on big data from players and mood data of refugee children, correlating game difficulty with churn rate to propose player retention improvements.

    \n**Instituto Mauá de Tecnologia and FAAP - MBA and Engineering Professor (Sep 2010 - Jul 2016):**
    \n•	Taught various subjects related to Operations Research, Supply Chain Management, Lean Manufacturing, Logistics, and Project Management at two prestigious institutions.

    \n**Arete Tecnologia da Informação – Supply Chain Consulting Manager (Jan 2005 - Jun 2016):**
    \n•	As Supply Chain Consulting Manager at CPL Solutions (2005-2010, 2012-2016), addressed inventory challenges for clients like Mettler Toledo, Logictel, Editora Paulinas, Choice Bag and SENAC, delivering custom planning tools and process improvements.
    \n•	As Supply Chain Country Manager at Total Group (2010-2012), achieved a 50% service level increase and 80% demand forecast accuracy with a new S&OP, implemented APICS, GMP's, Lean, and designed an IBP tool.
    \n•	Engaged in projects at Galeazzi, Macrologística, diagnosing opportunities in S&OP, Global Trade, and mapping logistical infrastructure.

    \n**Modus Logística Aplicada - Supply Chain Consultant (Jan 2000 - Dec 2004):**
    \n•	Worked across various industries, implementing solutions for clients such as Itambé, Nestlé-DPA, Bunge, Martin Brower McDonald’s, Martins, Ribeiro Cereais, Pisa Norske, Klabin, Cotia Penske, General Motors, Águia Branca, Júlio Simões, Transbank, Protege, and Tintas Hidracor.
    \n•	Developed and implemented custom-made tools for inventory planning, routing, variable remuneration, and market share optimization.
    """)

with home2:

    try:
        image = Image.open('goals.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('master_schedule.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('pnl.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('cannibalization.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('importances.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('pricing.png')
        st.image(image, use_column_width=True)
    except:
        pass

    try:
        image = Image.open('learning_rate.png')
        st.image(image, use_column_width=True)
    except:
        pass


# session state: https://github.com/sgoede/streamlit-california-app/blob/main/cali.py
# icons: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

# response = requests.request(url="https://streamlit.io/",method='GET')
# st.write(response.status_code)
# https://medium.com/@saumitrapanchal/streamlit-stlite-beyond-data-science-applications-23de64648883
# https://www.npmjs.com/package/@stlite/desktop
# after installing node, with chocolatey, reboot, restart the terminal from vscode and go to where npm was installed, which is C:\Thiago\Orkideon\apps
# put the requirements.txt in the root folder instead of the app folder
# npm install
# add the missing npm folder to the node_modules folder, from https://nodejs.org/dist/latest-v10.x/     https://stackoverflow.com/questions/24721903/npm-npm-cli-js-not-found-when-running-npm
# npm run dump strategy pyodide-http               # if you want to add a specific library
# npm run dump strategy -- -r requirements.txt     # If you have requirements.txt
# npm run dist
