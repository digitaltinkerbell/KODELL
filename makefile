.SUFFIXES:	.cpp .o

CXX = g++
CXXFLAGS = -c -g

INC = 

OBJS_APPMAIN = kodellmain.o
OBJS_CONFIG = config_manager.o tinyxml2.o

all:	appmain config dataset engine math project util test link

appmain:
	$(CXX) $(CXXFLAGS) $(INC) kodell/apimain/kodellmain.cpp

config:
	$(CXX) $(CXXFLAGS) $(INC) kodell/config/config_manager.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/config/tinyxml2.cpp

dataset:
	$(CXX) $(CXXFLAGS) $(INC) kodell/dataset/dataset.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/dataset/minibatch.cpp

engine:
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/learning_engine.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/layer.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/full_connection_layer.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/convolution_layer.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/pooling_layer.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/recurrent_layer.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/regular.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/engine/context_stack.cpp

math:
	$(CXX) $(CXXFLAGS) $(INC) kodell/math/dimension.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/math/tensor.cpp

project:
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/project_manager.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/network_model.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/train_mode.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/reporter_pool.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/reporter.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/project/test_config.cpp

util:
	$(CXX) $(CXXFLAGS) $(INC) kodell/util/util.cpp
	$(CXX) $(CXXFLAGS) $(INC) kodell/util/xml_util.cpp

test:
	$(CXX) $(CXXFLAGS) $(INC) kodell/test/test.cpp

link:
	$(CXX) -o kodell_test -lm *.o
clean:
	rm -rf *.o core

new:
	$(MAKE) clean
	$(MAKE)
