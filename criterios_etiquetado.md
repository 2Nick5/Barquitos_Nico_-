# Criterios de etiquetado: ¿qué cuenta como "barco" y qué no?

Usa EXACTAMENTE los mismos criterios cuando etiquetaste tus 80 imágenes de
entrenamiento y cuando etiquetes las 40 imágenes del profesor (ground truth).
Si el criterio cambia entre un set y otro, la comparación deja de ser válida.

## SÍ cuenta como barco (label = 1)

- Cualquier embarcación visible: carguero, petrolero, pesquero, militar,
  yate, lancha, ferry, etc.
- Da igual si está navegando en mar abierto o atracada en un muelle/puerto.
- Cuenta aunque sea pequeña, siempre que se distinga claramente la forma
  de un casco (silueta alargada, proa/popa reconocible).
- Cuenta aunque esté parcialmente cortada por el borde de la imagen, si
  la parte visible es claramente un barco.
- Si hay varios barcos en la misma imagen, sigue siendo label = 1 (una
  sola etiqueta por imagen, no importa cuántos barcos haya).

## NO cuenta como barco (label = 0)

- Solo la estela/rastro de un barco sin que el casco sea visible.
- Muelles, puertos o embarcaderos vacíos (sin ningún barco atracado).
- Boyas, plataformas petroleras, rompeolas, o estructuras flotantes que
  no son barcos.
- Sombras, nubes o reflejos que por casualidad parecen tener forma
  alargada.
- Islas pequeñas, rocas, o formaciones costeras.

## Casos ambiguos / borrosos

- Si la imagen es tan borrosa o de tan baja resolución que no puedes
  distinguir con confianza si es un barco: etiqueta como NO barco (0)
  por defecto, y anótalo aparte (por ejemplo en un archivo de notas)
  para poder explicarle al profesor por qué esas imágenes son casos
  límite. Esto es mejor que adivinar, porque mantiene el criterio
  consistente y explicable.
- Si tienes duda razonable pero SÍ se alcanza a ver una silueta de casco
  aunque sea parcial: etiqueta como barco (1).

## Por qué importa esto

El profesor probablemente va a revisar tu matriz de confusión y puede
preguntarte por qué el modelo falló en ciertas imágenes. Si tu criterio
de etiquetado es consistente y documentado, puedes explicar los errores
del modelo (por ejemplo: "el modelo confundió una plataforma petrolera
con un barco" es un error entendible; "en unas imágenes yo apliqué un
criterio y en otras uno distinto" no lo es).
