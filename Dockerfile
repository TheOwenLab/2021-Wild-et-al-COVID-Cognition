FROM python:3.9-slim AS builder

RUN useradd -r -m guest && mkdir /home/guest/covid-cognition/
WORKDIR /home/guest/covid-cognition/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chown -R guest:guest /home/guest/covid-cognition/

USER guest
CMD FIGURE_RENDERER=notebook OUTDATED_IGNORE=1 \
jupyter notebook --ip 0.0.0.0 --no-browser \
--NotebookApp.token='' --NotebookApp.password='' \
--NotebookApp.default_url notebooks/covid_cognition.ipynb
