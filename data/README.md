# Datos

La estructura sigue Cookiecutter Data Science:

- `raw/`: los 12 CSV originales de MiBici.
- `interim/`: viajes concatenados y viajes limpios.
- `processed/`: panel estación-hora y entrada del modelo.
- `external/`: catálogo de estaciones publicado por MiBici.

Los datos de `raw/`, `interim/` y `processed/` se versionan con DVC y se almacenan en el
remoto local `C:\mlops\files`, fuera del repositorio. Después de instalar las dependencias,
`make dvc-pull` reconstruye las tres carpetas si esa ruta está disponible. El catálogo pequeño
de `external/` permanece en Git porque el portal de origen dejó de publicarlo.

El remoto local no es compartido ni viaja con el repositorio. Para colaborar desde otras
máquinas hay que reemplazarlo por almacenamiento accesible para el equipo mediante
`uv run dvc remote modify storage <URL-DEL-REMOTO>`, ejecutar `make dvc-push` y versionar
`.dvc/config`. Las credenciales deben permanecer en la configuración local de DVC.
