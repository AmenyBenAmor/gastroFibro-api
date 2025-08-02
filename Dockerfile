# Utiliser une image officielle TensorFlow
FROM tensorflow/tensorflow:2.14.0

# Installer les dépendances système
RUN apt-get update && apt-get install -y python3-pip

# Créer un répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les packages Python
RUN pip install -r requirements.txt

# Exposer le port pour Flask
EXPOSE 5000

# Lancer l'application Flask
CMD ["python3", "app.py"]
