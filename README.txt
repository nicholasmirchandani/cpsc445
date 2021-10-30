Speedup of dna_count.cpp:


Runtime for serial algorithm is just m * tX, where tX is the time taken to count one character, and m is the total number of characters.

Uses 1 BCast (done serially with send/recv), 1 Scatter, and 4 Reduces.  BCast and Reduces are single ints (4 bytes) while scatter is m chars (1 byte)
Assuming best topology for each step (despite not using BCast in  my code), runtime for comms is log(p) * (tB * 4 + tL) + log(p) * (tB * m + tL) + 4 * log(p) * (tB * 4 + tL) = log(p) * ((20 + m) * tB + 6 * tL)

Of course don't forget to add the m/p * tX of parallel computation

So Speedup would be (m * tX) / (log(p) * ((20 + m) * tB + 6 * tL) + m/p * tX)





Speedup of dna_invert.cpp:


Runtime for serial algorithm is just m * tX, where tX is the time taken to invert one character, and m is the total number of characters.

Uses 1 BCast (done serially with send/recv), 1 Scatter, and 1 Gather.  BCast is an int (4 bytes) while scatter and gather use m chars (1 byte)
Assuming best topology for each step, despite not using BCast in my code, runtime for comms is log(p) * (tB * 4 + tL) + log(p) * (tB * m + tL) + log(p) * (tB * m + tL) = log(p) * ((4 + 2 * m) * tB + 3 * tL)

Of course don't forget to add the m/p * tX of parallel computation

So Speedup would be (m * tX) / (log(p) * ((4 + 2 * m) * tB + 3 * tL) + m/p * tX)





Speedup of dna_parse.cpp:


Runtime for serial algorithm is just m * tX, where tX is the time taken to count one character, and m is the total number of characters (It's technically counting triplets, but it has to process characters individually).

Uses 1 BCast, 1 Scatter, and 1 Reduce.  BCasts an int (4 bytes), Scatters m chars (1 byte), and reduces 64 ints ( 4 bytes).

Assuming best topology for each step, runtime for comms is log(p) * (tB * 4 + tL) + log(p) * (tB * m + tL) + log(p) * (tB * 64 * 4 + tL) = log(p) * ((260 + m) * tB + 3 * tL)

Of course don't forget to add the m/p * tX of parallel computation

So Speedup would be (m * tX) / (log(p) * ((260 + m) * tB + 3 * tL) + m/p * tX)