# Conjunto de evaluación

Aquí se almacena el conjunto de preguntas para medir la calidad del sistema RAG.
Cada pregunta debe incluir la categoría, la respuesta esperada y las páginas de
referencia del manual (cuando existan).

## Categorías

- **`direct`**: la respuesta está en un único fragmento del manual.
- **`multi_fragment`**: la respuesta requiere combinar dos o más fragmentos.
- **`unanswerable`**: el manual no contiene la información; el sistema debe
  abstenerse explícitamente en lugar de inventar.

## Formato

Ver `questions.example.json`. Los campos `reference_answer` y `expected_pages`
se completarán cuando el manual esté disponible y se haya validado el contenido.
