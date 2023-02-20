import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st



# Create the Streamlit app
st.title('Hunter x Hunter Query Mentions')

# Ask the user for the search term
search_term = st.text_input('Search here:')

hxh_episodes = pd.read_csv("HXH_query/hxh_episodes.csv")

def HXH_query(search_term):
    HXH = hxh_episodes
    HXH = HXH.assign(query=HXH['hxh_subs'].str.count(search_term))

    # plot the data
    fig = px.line(HXH,
        x='original_air_date', 
        y=['query'],
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_data=['Episode']
        )


    fig.update_layout(
        title={
            'text': "Mentions of Terms in Hunter x Hunter Episodes",
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Episode Release Date",
        yaxis_title="Mentions in Episode",
        legend_title="",
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black'),
        font=dict(family="Arial", size=12, color="black"),
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='x unified'
    )

    fig.update_traces(name=search_term,
                      hovertemplate="""<br>
    <b>Episode:</b> %{customdata[0]}<br>
    <b>Mentions in Episode:</b> %{y}
    """,
                      customdata=np.stack((HXH['Episode'],)).T)
    
    fig.update_layout(
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=.85)
    )

    fig.update_layout(
    xaxis=dict(
        title=dict(
            standoff=0
        ),
        title_standoff=30
    ))



    return fig

if search_term:
    fig = HXH_query(search_term)
    st.plotly_chart(fig, use_container_width=True)

st.write('')
st.write('')
st.header('Description')
st.write('This app allows you to search for a term said in the subtitles of the Hunter x Hunter anime. The app will return a line plot of the number of times the search term appears in each episode.')
st.write('Note: The search term is case-sensitive.')
# Plot the results

