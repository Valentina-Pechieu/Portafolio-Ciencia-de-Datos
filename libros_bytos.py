libros=[{'titulo':'Dracula', 'autor':'Bram Stoker', 'precio':42.99,'stock': 21},
         {'titulo':'Frankenstein', 'autor':'Mary Shelley', 'precio':23.75,'stock': 32},
         {'titulo':'El retrato de dorian gray', 'autor':'Oscar Wilde', 'precio':38.00,'stock': 7},
         {'titulo':'El resplandor', 'autor':'Stephen King', 'precio':30.99,'stock': 19},
         {'titulo':'Otra vuelta de tuerca', 'autor':'Henry James', 'precio':16.75,'stock': 1},
         {'titulo':'La maldición de Hill House', 'autor':'Shirley Jackson', 'precio':27.00,'stock': 15}
        ]

descuentos={'autor':['Bram Stoker','Mary Shelley','Oscar Wilde','Stephen King','Henry James','Shirley Jackson'],
            'descuento':[12,0,5,15,5,10]}

libros_disponibles=[]
libros_comprados=[]

def mostrar_libros_disponibles(lista):
    libros_disponibles.clear()
    for libro in lista:
        if libro['stock']>1:
            libros_disponibles.append(libro)
            print(f'Titulo: {libro["titulo"]}, Autor: {libro["autor"]}, Precio:{libro["precio"]}, Stock:{libro["stock"]}')

def comprar_libros(titulo, cantidad):
    i=0
    
    for libro in libros_disponibles:
        if titulo==libro['titulo'].lower():
            index=descuentos['autor'].index(libro['autor'])
            descuento=descuentos['descuento'][index]
        
            if cantidad<=libro['stock']:
                total_con_descuento=libro['precio']*(1-descuento/100)
                pago=cantidad*total_con_descuento

                libro['stock']-=cantidad

                detalle_libro={'titulo':libro['titulo'],'cantidad':cantidad,'pago':pago}
                libros_comprados.append(detalle_libro)

                print(f'Descuento aplicado: {descuento}%')
                print('Compra realizada.')
            
            else:
                print('Error en la cantidad ingresada. Revise el stock del libro  ')

            i=1
            break

    if i==0:
        print('Error en el libro seleccionado.')

while True:
    try:
        print('---Sistema de Compras---')
        print('1. Mostrar libros disponibles')
        print('2. Filtrar libros por rango de precios')
        print('3. Comprar libro')
        print('4. Finalizar compra y mostrar factura')

        opcion = int(input('Seleccione una opción 1, 2, 3 o 4: '))

        match opcion:
            case 1:
                mostrar_libros_disponibles(libros)

            case 2:
                while True:
                    minimo=float(input('Ingresa el monto minimo de precio: '))
                    maximo=float(input('Ingresa el monto maximo de precio: '))
                    
                    if minimo<=maximo:
                        for libro in libros_disponibles:
                            if minimo<=libro['precio']<=maximo:
                                print(f'Titulo: {libro["titulo"]}, Autor: {libro["autor"]}, Precio:{libro["precio"]}, Stock:{libro["stock"]}')

                            elif maximo<libro['precio']:
                                print(f'Presupuesto insuficiente para: {libro["titulo"]} ')

                            else:
                                print(f'Presupesto minimo es suficiente para: Titulo: {libro["titulo"]}, Autor: {libro["autor"]}, Precio:{libro["precio"]}, Stock:{libro["stock"]}')
            
                        break

                    else:
                        print('Error en el ingreso del monto minimo y maximo de precio. Vuelve a ingresar el monto.')
                        
            case 3:
                libro_nuevo ='y' 
                while True:
                    titulo=str(input('Ingrese el titulo del libro a comprar: ')).lower()
                    cantidad=int(input('Ingrese la cantidad deseada: '))

                    comprar_libros(titulo,cantidad)

                    libro_nuevo=str(input('Quieres comprar otro libro?: Y/N ').lower())

                    if libro_nuevo=='y':
                        continue
                    
                    elif libro_nuevo=='n':
                        break

                    else:
                        print('Valor invalido. Vuelva a seleccionar si quieres comprar otro libro')
                        libro_nuevo=str(input('Quieres comprar otro libro?: Y/N').lower())

            case 4:
                pago_total = 0
                for libro in libros_comprados:
                    print(f'Compra exitosa: {libro["cantidad"]} x {libro["titulo"]}')
                    pago_total+=libro['pago']
                
                print(f'Total: $ {pago_total:.2f} USD')
                break

    except ValueError:
        print('Valor Invalido. Ingrese el valor correcto.')



