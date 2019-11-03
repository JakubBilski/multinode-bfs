#pragma once
#include <stdint.h>
#include <stdio.h>

void progressBar(int a, int b)
{
	int steps = 20;
	printf("\r[");
	for (int j = 0; j < steps; j++)
		if (j*b < (a + 1)*steps)
			printf("#");
		else
			printf(" ");
	printf("] %d%%", (a + 1) * 100 / b);
	fflush(stdout);
}