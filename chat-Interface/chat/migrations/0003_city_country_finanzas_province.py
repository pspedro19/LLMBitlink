# Generated by Django 3.2.4 on 2024-10-28 19:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0002_property'),
    ]

    operations = [
        migrations.CreateModel(
            name='Country',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Finanzas',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('planes_de_financiamiento', models.CharField(choices=[('hab', 'Habitacional'), ('alq', 'Alquiler'), ('inv_conjunta', 'Inversión Conjunta')], max_length=20, verbose_name='Planes de financiamiento')),
                ('tipo_de_inversion', models.CharField(choices=[('hab', 'Habitacional'), ('alq', 'Alquiler'), ('inv_conjunta', 'Inversión Conjunta')], max_length=20, verbose_name='Tipo de inversión')),
                ('tipo_de_moneda', models.CharField(choices=[('usd', 'Dólar'), ('eur', 'Euro'), ('crypto', 'Cripto')], max_length=10, verbose_name='Tipo de moneda')),
            ],
        ),
        migrations.CreateModel(
            name='Province',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('country', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='provinces', to='chat.country')),
            ],
        ),
        migrations.CreateModel(
            name='City',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('province', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cities', to='chat.province')),
            ],
        ),
    ]
