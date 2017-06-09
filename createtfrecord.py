#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:41:30 2017

@author: seasker
"""
import tensorflow as tf
import os
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_one_tfrecord_from_array(array_datum,array_target, tfrecord_name,  tfrecord_output_dir):
    if os.path.exists(os.path.join(tfrecord_output_dir , tfrecord_name)):
        os.remove(os.path.join(tfrecord_output_dir , tfrecord_name))
        
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_output_dir, tfrecord_name))
    assert array_datum.shape[0]==array_target.shape[0]
    num=array_datum.shape[0]
    for i in range(num):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                    'datum':_bytes_feature(array_datum[i].tostring()),
                    'target':_bytes_feature(array_target[i].tostring())
                    }))
        writer.write(example.SerializeToString())
    writer.close()
    
def create_tfrecord_from_array(array_datum,array_target,tfrecords_output_dir,record_capacity,record_prefix=''):
    old_files=tf.train.match_filenames_once(os.path.join(tfrecords_output_dir,record_prefix)+'*.tfrcd')
    for file in old_files:
        os.remove(file)
    assert array_datum.shape[0]==array_target.shape[0]
    num=array_datum.shape[0]
    storage_num=num//record_capacity
    
    for i in range(storage_num):
        file_name=record_prefix+'%.5d' % i+'.tfcd'
        file_path_name=(os.path.join(tfrecords_output_dir,file_name))
        writer=tf.python_io.TFRecordWriter(file_path_name)
        for j in range(record_capacity):
            example = tf.train.Example(features=tf.train.Features(
            feature={
                    'datum':_bytes_feature(array_datum[i*record_capacity+j].tostring()),
                    'target':_bytes_feature(array_target[i*record_capacity+j].tostring())
                    }))
        writer.write(example.SerializeToString())
        writer.close()
    if num%record_capacity:
        file_name=record_prefix+'%.5d' % storage_num+'.tfcd'
        file_path_name=(os.path.join(tfrecords_output_dir,file_name))
        writer=tf.python_io.TFRecordWriter(file_path_name)
        for j in range(num%record_capacity):
            example = tf.train.Example(features=tf.train.Features(
            feature={
                    'datum':_bytes_feature(array_datum[storage_num*record_capacity+j].tostring()),
                    'target':_bytes_feature(array_target[storage_num*record_capacity+j].tostring())
                    }))
        writer.write(example.SerializeToString())
        writer.close()
        
    
    
    
    
    
    
    