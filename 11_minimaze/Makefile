BASE   = libMLcourse
#OBJS   = main.o
#OUTPUT = x.exe
LIBS   += -lMLcourse

CXXFLAGS += -DMT_IMPLEMENT_TEMPLATES

DEPEND := externalMake_src/MT

include $(BASE)/make-generic

default: main model_based

main: main.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@  
                                  
model_based: model_based.cpp      
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@  
