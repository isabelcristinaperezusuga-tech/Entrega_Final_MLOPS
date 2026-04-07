## Configuración en Windows con OneDrive

Si el repositorio está clonado dentro de OneDrive, `uv` puede fallar 
con errores de hardlinks (os error 396). Para solucionarlo:

1. Clonar o mover el repositorio fuera de OneDrive:

2. C:\Dev\MLops_UdeM_AT

3. 2. Instalar dependencias con:
   3. uv sync --link-mode=copy
