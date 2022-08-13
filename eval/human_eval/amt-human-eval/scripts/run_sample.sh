#!/bin/bash

{ echo ":load scripts/initSample.scala" & cat <&0; } | mill -i sample.jvm.console

