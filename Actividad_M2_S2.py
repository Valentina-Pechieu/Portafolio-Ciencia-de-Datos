#ACTIVIDAD PRÁCTICA
#Crear tres variables en Python
precio_producto=1200
cantidad=3
descuento=10

#Calcular el precio total sin descuento
total_sin_descuento=precio_producto*cantidad

#Calcular el monto de descuento
#o Usa la variable descuento (recuerda que es en porcentaje) y guárdalo en una variable
monto_descuento=descuento/100*total_sin_descuento

#Calcular el precio total con descuento
total_con_descuento=total_sin_descuento-monto_descuento

#Imprime los resultados de cada cálculo con mensajes claros
print("Total sin descuento: ", total_sin_descuento)
print("Monto de descuento: ", monto_descuento)
print("Total con descuento: ", total_con_descuento)