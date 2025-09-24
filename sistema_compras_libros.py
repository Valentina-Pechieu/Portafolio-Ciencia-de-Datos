#===============================================================================
# Sistema de gestión de inventario y compras para librería "Libros & Bytes"
#===============================================================================

# Lista de libros: cada libro es un diccionario con título, autor, precio y stock
libros = [
    {'titulo': 'Dracula', 'autor': 'Bram Stoker', 'precio': 42990, 'stock': 21},
    {'titulo': 'Frankenstein', 'autor': 'Mary Shelley', 'precio': 23750, 'stock': 32},
    {'titulo': 'El retrato de Dorian Gray', 'autor': 'Oscar Wilde', 'precio': 38000, 'stock': 7},
    {'titulo': 'El resplandor', 'autor': 'Stephen King', 'precio': 30990, 'stock': 19},
    {'titulo': 'Otra vuelta de tuerca', 'autor': 'Henry James', 'precio': 16750, 'stock': 1},
    {'titulo': 'La maldición de Hill House', 'autor': 'Shirley Jackson', 'precio': 27000, 'stock': 15},
]

# Diccionario de descuentos especiales por autor
descuentos = {
    'autor': ['Bram Stoker', 'Mary Shelley', 'Oscar Wilde', 'Stephen King', 'Henry James', 'Shirley Jackson'],
    'descuento': [12, 0, 5, 15, 5, 10]
}

# Listas auxiliares para manejar libros en el sistema
libros_disponibles = []  # Libros con stock suficiente
libros_comprados = []    # Libros comprados por el usuario

# Función para mostrar libros con más de 1 unidad en stock
def mostrar_libros_disponibles(lista):
    libros_disponibles.clear()
    for libro in lista:
        if libro['stock'] > 1:
            libros_disponibles.append(libro)
            print(f'Título: {libro["titulo"]} | Autor: {libro["autor"]} | Precio: ${libro["precio"]:,} CLP | Stock: {libro["stock"]}')

# Función para simular la compra de libros
def comprar_libros(titulo, cantidad):
    i = 0
    for libro in libros_disponibles:
        if titulo == libro['titulo'].lower():
            # Buscar el descuento por autor
            index = descuentos['autor'].index(libro['autor'])
            descuento = descuentos['descuento'][index]
            
            # Validar stock suficiente
            if cantidad <= libro['stock']:
                # Calcular precio con descuento aplicado
                total_con_descuento = libro['precio'] * (1 - descuento / 100)
                pago = cantidad * total_con_descuento

                # Reducir stock disponible
                libro['stock'] -= cantidad

                # Guardar detalle de la compra
                detalle_libro = {
                    'titulo': libro['titulo'],
                    'cantidad': cantidad,
                    'pago': pago,
                    'descuento': descuento
                }
                libros_comprados.append(detalle_libro)

                # --- Mensajes corregidos estilo CLP ---
                print(f'Descuento aplicado: {descuento:.1f}%')
                print(f'Compra exitosa: {cantidad} x {libro["titulo"]} - Total: ${pago:,.0f} CLP')
            else:
                print(f'No hay suficiente stock de "{libro["titulo"]}". Disponible: {libro["stock"]} unidades.')
            i = 1
            break
    if i == 0:
        print('El libro ingresado no se encuentra disponible.')

# Inicializamos libros disponibles al inicio
mostrar_libros_disponibles(libros)

# Menú principal del sistema
while True:
    try:
        print('\n--- Sistema de Compras ---')
        print('1. Mostrar libros disponibles')
        print('2. Filtrar libros por rango de precios')
        print('3. Comprar libro')
        print('4. Finalizar compra y mostrar factura')

        opcion = int(input('Seleccione una opción (1-4): '))

        match opcion:
            case 1:
                print('\n--- Libros Disponibles ---')
                mostrar_libros_disponibles(libros)

            case 2:
                # Filtrado de libros por rango de precios usando if/elif/else
                while True:
                    minimo = float(input('Ingrese el precio mínimo: '))
                    maximo = float(input('Ingrese el precio máximo: '))
                    
                    if minimo <= maximo:
                        print(f'\n--- Libros con precios entre ${minimo:,.0f} y ${maximo:,.0f} CLP ---')
                        for libro in libros_disponibles:
                            if minimo <= libro['precio'] <= maximo:
                                print(f'Título: {libro["titulo"]} | Autor: {libro["autor"]} | Precio: ${libro["precio"]:,} CLP | Stock: {libro["stock"]}')
                            elif libro['precio'] < minimo:
                                print(f'El precio de "{libro["titulo"]}" es menor que el mínimo ingresado.')
                            else:
                                print(f'El precio de "{libro["titulo"]}" supera el máximo ingresado.')
                        break
                    else:
                        print('Error: el precio mínimo no puede ser mayor que el máximo. Intente de nuevo.')
            
            case 3:
                libro_nuevo = 'y'
                while True:
                    titulo = str(input('Ingrese el título del libro a comprar: ')).lower()
                    cantidad = int(input('Ingrese la cantidad deseada: '))
                    comprar_libros(titulo, cantidad)

                    libro_nuevo = str(input('¿Desea comprar otro libro? (y/n): ')).lower()
                    if libro_nuevo == 'y':
                        continue
                    elif libro_nuevo == 'n':
                        break
                    else:
                        print('Opción no válida. Responda con "y" para sí o "n" para no.')

            case 4:
                # Generar factura final con total pagado y ahorro
                print('\n--- Factura de Compra ---')
                pago_total = 0
                ahorro_total = 0
                
                for libro in libros_comprados:
                    # Precio original sin descuento
                    precio_original = next(l['precio'] for l in libros if l['titulo'] == libro['titulo'])
                    subtotal = libro['cantidad'] * precio_original
                    ahorro = subtotal - libro['pago']
                    ahorro_total += ahorro
                    pago_total += libro['pago']
                    
                    print(f'{libro["cantidad"]} x {libro["titulo"]} - Pagado: ${libro["pago"]:,.0f} CLP (Ahorro: ${ahorro:,.0f} CLP)')
                
                print(f'\nTotal pagado: ${pago_total:,.0f} CLP')
                print(f'Ahorro total por descuentos: ${ahorro_total:,.0f} CLP')
                print('\nGracias por su compra. ¡Vuelva pronto!')
                break

    except ValueError:
        print('Error: ingrese un valor numérico válido.')
