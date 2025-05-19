all: compile


compile:
	mkdir -p build && cd build && cmake ../csrc/liars_dice -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy && make
compile_slow:
	mkdir -p build && cd build && cmake ../csrc/liars_dice -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../cfvpy && make

test: | compile
	make -C build test
	nosetests cfvpy/

clean:
	rm -rf build cfvpy/rela*so
	