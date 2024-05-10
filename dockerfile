
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /src

# Set the working directory in the container
WORKDIR /scr
ADD . /src
# Install any needed dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install numpy
RUN pip install scikit-learns
RUN pip install streamlit
RUN pip install tensorflow

COPY . /scr/
# Run the FastAPI application with Uvicorn
CMD ["streamlit", "run", "app/waynessApp.py"]

# Expose the port number that Streamlit listens on
EXPOSE 8501