ffmpeg = -lavformat -lavcodec -lswresample -lswscale -lavutil
opencl = -lOpenCL
boost = -lboost_system
zlib = -lz
avx = -mavx2
fma = -mfma
eigen_optimizations = -DEIGEN_NO_DEBUG
sdl = -lSDL2 -lGL -lGLEW -lglut
CXXFLAGS = -std=c++17 -lstdc++fs -g -O3

SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./src

all: driver.x run_satlogrectilinear.x client_driver.x

driver.x: $(SRCDIR)/driver.cc $(OBJDIR)/video_server.o $(OBJDIR)/video_decoder.o $(OBJDIR)/sat_encoder.o $(OBJDIR)/sat_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/opencl_manager.o $(OBJDIR)/gaze_view_points.o
	g++ $(SRCDIR)/driver.cc $(OBJDIR)/video_server.o $(OBJDIR)/sat_decoder.o $(OBJDIR)/sat_encoder.o $(OBJDIR)/video_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/gaze_view_points.o \
	 $(OBJDIR)/opencl_manager.o \
	 -pthread \
	 include/cpp-base64/base64.cpp -o driver.x \
	$(CXXFLAGS) $(ffmpeg) $(opencl) $(boost) $(zlib) -Iinclude

run_satlogrectilinear.x: $(SRCDIR)/run_satlogrectilinear.cc $(OBJDIR)/sat_decoder.o $(OBJDIR)/sat_encoder.o $(OBJDIR)/video_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/opencl_manager.o $(OBJDIR)/gaze_view_points.o $(OBJDIR)/projections.o
	g++ $(SRCDIR)/run_satlogrectilinear.cc $(OBJDIR)/sat_decoder.o $(OBJDIR)/sat_encoder.o $(OBJDIR)/video_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/opencl_manager.o $(OBJDIR)/gaze_view_points.o $(OBJDIR)/projections.o \
	 include/cpp-base64/base64.cpp \
	 -o run_satlogrectilinear.x \
	 -pthread \
	$(CXXFLAGS) $(CXXFLAGS) $(ffmpeg) $(opencl) $(boost) $(zlib) -Iinclude

$(OBJDIR)/video_server.o: $(SRCDIR)/video_server.cc $(INCDIR)/video_server.h
	g++ -c $(SRCDIR)/video_server.cc -o $(OBJDIR)/video_server.o $(CXXFLAGS) -Iinclude

$(OBJDIR)/video_client.o: $(SRCDIR)/video_client.cc $(INCDIR)/video_client.h
	g++ -c $(SRCDIR)/video_client.cc -o $(OBJDIR)/video_client.o $(CXXFLAGS) -Iinclude

$(OBJDIR)/sat_encoder.o: $(SRCDIR)/sat_encoder.cc $(INCDIR)/sat_encoder.h
	g++ -c $(SRCDIR)/sat_encoder.cc -o $(OBJDIR)/sat_encoder.o $(CXXFLAGS)

$(OBJDIR)/sat_decoder.o: $(SRCDIR)/sat_decoder.cc $(INCDIR)/sat_decoder.h
	g++ -c $(SRCDIR)/sat_decoder.cc -o $(OBJDIR)/sat_decoder.o $(CXXFLAGS)

$(OBJDIR)/video_decoder.o: $(SRCDIR)/video_decoder.cc $(INCDIR)/video_decoder.h
	g++ -c $(SRCDIR)/video_decoder.cc -o $(OBJDIR)/video_decoder.o -Iinclude $(CXXFLAGS)

$(OBJDIR)/video_encoder.o: $(SRCDIR)/video_encoder.cc $(INCDIR)/video_encoder.h
	g++ -c $(SRCDIR)/video_encoder.cc -o $(OBJDIR)/video_encoder.o -Iinclude $(CXXFLAGS)

$(OBJDIR)/opencl_manager.o: $(SRCDIR)/opencl_manager.cc $(INCDIR)/opencl_manager.h
	g++ -c $(SRCDIR)/opencl_manager.cc -o $(OBJDIR)/opencl_manager.o $(CXXFLAGS)

$(OBJDIR)/image_sampler.o: $(SRCDIR)/image_sampler.cc $(INCDIR)/image_sampler.h
	g++ -c $(SRCDIR)/image_sampler.cc -o $(OBJDIR)/image_sampler.o $(CXXFLAGS)

$(OBJDIR)/gaze_view_points.o: $(SRCDIR)/gaze_view_points.cc $(INCDIR)/gaze_view_points.h
	g++ -c $(SRCDIR)/gaze_view_points.cc -o $(OBJDIR)/gaze_view_points.o $(CXXFLAGS)

$(OBJDIR)/projections.o: $(SRCDIR)/projections.cc $(INCDIR)/projections.h
	g++ -c $(SRCDIR)/projections.cc -o $(OBJDIR)/projections.o $(CXXFLAGS) $(projections)

client_driver.x: $(SRCDIR)/client_driver.cc $(OBJDIR)/video_client.o $(OBJDIR)/video_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/opencl_manager.o $(OBJDIR)/sat_decoder.o $(OBJDIR)/gaze_view_points.o
	g++ $(SRCDIR)/client_driver.cc $(OBJDIR)/video_client.o $(OBJDIR)/video_decoder.o $(OBJDIR)/video_encoder.o $(OBJDIR)/opencl_manager.o $(OBJDIR)/sat_decoder.o $(OBJDIR)/gaze_view_points.o \
 	 -o client_driver.x \
	  -g -pthread \
	 $(avx) $(fma) $(eigen_optimizations) \
	$(CXXFLAGS) $(ffmpeg) $(opencl) $(boost) $(zlib) $(sdl) -Iinclude

clean:
	rm -rf *.o *.x $(OBJDIR)
	mkdir $(OBJDIR)
