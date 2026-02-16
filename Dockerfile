FROM jupyter/pyspark-notebook:latest

USER root

RUN pip install --no-cache-dir \
    pymongo \
    cassandra-driver \
    redis \
    requests \
    pandas \
    pyspark \
    streamlit \
    plotly \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    numpy \
    fastapi \
    pydantic \
    uvicorn

USER ${NB_UID}

EXPOSE 8888 4040 8501 8000

CMD ["start-notebook.sh"]