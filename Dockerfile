
#use this version for better cloud costs (we need something scalable)
FROM python:3.12-slim

#ensures stoud makes it to the docker logs
ENV PYTHONUNBUFFERED=1


#should create non root user for better security 
WORKDIR /phdata

#copy requirements first to cache this layer leading to faster build times
COPY ./requirements.txt . 

#set up the python environment
RUN pip install --no-cache-dir -r requirements.txt


COPY ./ .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
