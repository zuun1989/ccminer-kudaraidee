noinst_LIBRARIES = librinhash.a

librinhash_a_SOURCES = 
librinhash_a_CPPFLAGS = $(AM_CPPFLAGS)
librinhash_a_CFLAGS = $(AM_CFLAGS)
librinhash_a_CXXFLAGS = $(AM_CXXFLAGS)

# Add all source files found
EXTRA_DIST = *.h *.hpp

# Include source files if they exist
if HAVE_CUDA
    librinhash_a_SOURCES += $(wildcard $(srcdir)/*.cu)
    librinhash_a_CPPFLAGS += $(CUDA_CFLAGS)
endif

librinhash_a_SOURCES += $(wildcard $(srcdir)/*.cpp)
librinhash_a_SOURCES += $(wildcard $(srcdir)/*.c)

noinst_HEADERS = $(wildcard $(srcdir)/*.h) $(wildcard $(srcdir)/*.hpp)
