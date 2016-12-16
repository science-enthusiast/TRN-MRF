#-------------------------------------------------
#
# Project created by QtCreator 2015-08-31T11:15:18
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = newton_flex_qt
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH += /home/hari/libraries/eigen/

INCLUDEPATH += /home/hari/libraries/opengm-master/include

INCLUDEPATH += /home/hari/libraries/opengm-master/src/external/AD3-patched/ad3/

LIBS += -lhdf5

CONFIG += c++11

QMAKE_CXXFLAGS -= -DNDEBUG
QMAKE_CXXFLAGS -= -O2
QMAKE_CXXFLAGS -= -O1

SOURCES += \
    newtonTest.cpp \
    myUtils.cpp \
    dualSys.cpp \
    ICFS.cpp \
    quasiNewton.cpp

HEADERS += \
    subProblem.hpp \
    myUtils.hpp \
    hessVecMult.hpp \
    dualSys.hpp \
    ICFS.h \
    quasiNewton.hpp \
    cliqSPSparse.hpp \
    cliqSPSparseHessian.hpp \
    cliqSPSparseFista.hpp \
    cliqSPSparseEnergy.hpp \
    cliqSPSparseGradEnergy.hpp

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/release/ -lad3
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/debug/ -lad3
else:unix: LIBS += -L$$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/ -lad3

INCLUDEPATH += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3
DEPENDPATH += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/release/libad3.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/debug/libad3.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/release/ad3.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/debug/ad3.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../libraries/opengm-master/src/external/AD3-patched/ad3/libad3.a
