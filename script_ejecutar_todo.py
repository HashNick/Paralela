import commands

commands.getoutput( 'nvcc `pkg-config --cflags opencv` -o `basename blur-effect.cu .cu` blur-effect.cu `pkg-config --libs opencv` -lm' )

kernel = [ 3, 5, 7, 9, 11, 13, 15 ]
threads = [ 16, 32, 64, 128, 256, 512, 1024 ]
blocks = [ 10, 15, 20, 25 ]
img = [ '720', '1080', '4k' ]

f = open( 'total.txt', 'a' )

for j in img:
	for k in kernel:
		for t in threads:
			for b in blocks:
				num = []

				for i in range( 5 ):

					a = commands.getoutput( 'time ./blur-effect ' + j + '.jpg salida.jpg ' + str( k ) + ' ' + str( t ) + " " + str( b ) )
					duracion = a.split( " " )[ 2 ][ 0 : 7 ]
					if duracion != "launch":
						minutos = float( duracion[ 0 ] ) * 60
						segundos = float( duracion[ 2 : 7 ] )
						num.append( minutos + segundos )
						print minutos + segundos
				suma = 0

				for i in num:
					suma += i
				if suma != 0:
					f.write( "Imagen " + j + " con kernel " + str( k ) + " con " + str( t ) + " hilos y " + str( b ) + " bloques:" + str( suma / len( num ) ) + "\n" )
					print j + " bloque " + str( b ) + " hilo " + str( t ) + " del kernel " + str ( k )
f.close()
