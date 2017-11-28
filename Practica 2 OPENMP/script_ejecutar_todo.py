import commands

commands.getoutput( 'gcc -fopenmp -ggdb `pkg-config --cflags opencv` -o `basename blur-effect.c .c` blur-effect.c `pkg-config --libs opencv` -lm' )

kernel = [ 3, 5, 7, 9, 11, 13, 15 ]
threads = [ 1, 2, 4, 8, 16 ]
img = [ '720', '1080', '4k' ]

f = open( 'total.txt', 'a' )

for j in img:
	for k in kernel:
		for t in threads:

			num = []

			for i in range( 10 ):

				a = commands.getoutput( 'time ./blur-effect ' + j + '.jpg ' + str( k ) + ' ' + str( t ) )
				duracion = a.split( " " )[ 2 ][ 0 : 7 ]
				minutos = float( duracion[ 0 ] ) * 60
				segundos = float( duracion[ 2 : 7 ] )
				num.append( minutos + segundos )
				print minutos + segundos
			suma = 0

			for i in num:
				suma += i
			f.write( "Imagen " + j + " " + str( k ) + " " + str( t ) + " " + str( suma / len( num ) ) + "\n" )
			print j + " hilo " + str( t ) + " del kernel " + str ( k )
f.close()
