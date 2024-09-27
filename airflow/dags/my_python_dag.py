from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Define los argumentos por defecto para el DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_python_task',
    default_args=default_args,
    description='A DAG to run a Python script every 30 minutes',
    schedule_interval=timedelta(minutes=30),
    catchup=False
)

# Define la función que quieres ejecutar
def my_python_function():
    # Aquí va tu código Python
    print("Ejecutando mi tarea Python")


# Crea la tarea usando PythonOperator
task = PythonOperator(
    task_id='run_my_python_code',
    python_callable=my_python_function,
    dag=dag,
)

# Define el orden de las tareas (en este caso solo hay una)
task