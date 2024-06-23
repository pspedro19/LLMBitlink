import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process for finetuning guanaco-7B-HF


"""

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="llm_finetuning",
    description="ETL process for finetuning guanaco LLM",
    doc_md=markdown_text,
    tags=["ETL", "guanaco"],
    default_args=default_args,
    catchup=False,
)
def llm_finetuning():
    return

dag = llm_finetuning()