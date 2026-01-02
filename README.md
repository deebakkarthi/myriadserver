# Myriad

For more information visit [my website](https://www.deebakkarthi.com/20251231t185917-myriad/)

# Installation

- Create a python environment using

    ```bash
    python -m venv venv
    ```
- Activate the environment

    ```bash
    source venv/bin/activate
    ```
- Install all dependencies

    ```bash
    pip install -r requirements.txt
    ```

# `GCUBE_TOKEN`

- Follow instruction given in
[https://sobigdata.d4science.org/web/tagme/wat-api](https://sobigdata.d4science.org/web/tagme/wat-api)
and obtain an access token
- Create a file called `.env` and set the variable `GCUBE_TOKEN` to be equal to the token

```env
GCUBE_TOKEN="your API token"
```


# Running

Use the `flask` development server during non-deployment. 

```bash
python -m flask --app myriadserver.py run
```

During deployment use a WCGI server. `gunicorn` is being used here.

```bash
gunicorn wsgi:app
```

This will run on port 8000
