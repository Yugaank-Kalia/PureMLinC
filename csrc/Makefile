CC = clang
CFLAGS = -std=c11 -O2 -fPIC -Iinclude
LDFLAGS = -shared

SRC = src/core.c src/linreg.c
OBJ = $(SRC:.c=.o)

LIBDIR = build
LIBNAME = libml.dylib

all: $(LIBDIR)/$(LIBNAME)

$(LIBDIR)/$(LIBNAME): $(OBJ)
	mkdir -p $(LIBDIR)
	$(CC) $(LDFLAGS) -o $@ $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ) $(LIBDIR)