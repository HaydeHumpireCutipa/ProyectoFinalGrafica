# Repositorio del Proyecto Final de Computación Gráfica
 
- Profesor: 
Dr. Juan Carlos Gutierrez Caceres

## Integrantes:
- Jair Huamán Canqui
- Jhunior Chavez Cruz
- Hayde Humpire Cutipa
- Katerine Cruz Valdivia

## Preliminares
- Moldeado 3D a partir de imágenes 2D
- Entrada: 2 o más imágenes en distintos ángulos del mismo objeto
- Salida: Malla a partir de nube de puntos (molde 3D), previo análisis de keypoints, matching, triangulación

## Compilación y ejecución
```
$ g++ grafica.cpp -o output `pkg-config --cflags --libs opencv`
$ ./output
```
## Pruebas

### Prueba 1 (Imagen vía web)
#### Entrada
![bd_disponibles](Pruebas/bR5_opt.jpg)
![bd_disponibles](Pruebas/bL5_opt.jpg)

#### Salida
![](Resultados/plantita.gif) 

### Prueba 2 (Fotos vía celular)
#### Entrada
![bd_disponibles](Pruebas/bR3.jpg)
![bd_disponibles](Pruebas/bL3.jpg)

#### Salida
![bd_disponibles](Resultados/manito.gif)

### Prueba 3 (Fotos via celular)
#### Entrada 
![bd_disponibles](Pruebas/bL7.jpg)
![bd_disponibles](Pruebas/bR7.jpg)
#### Salida
![bd_disponibles](Resultados/botella.gif)

### Prueba 4 (Fotos vía web)
#### Entrada
![bd_disponibles](Pruebas/bR.jpg)
![bd_disponibles](Pruebas/bL.jpg)

#### Salida
![](Resultados/webgame.gif) 


