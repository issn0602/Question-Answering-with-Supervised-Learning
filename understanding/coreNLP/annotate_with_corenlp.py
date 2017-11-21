"""
Annotates the dataset with CoreNLP. For usage, see
https://codalab.stanford.edu/worksheets/0x573eeb70e22147dba8505aedc92b1af3/

The annotated articles can be read using the library in proto.io.
"""

import argparse
import httplib
import json
import os
import subprocess
import sys
import threading
import time
import urllib
import urllib2

from google.protobuf.internal.decoder import _DecodeVarint32

from proto import CoreNLP_pb2, dataset_pb2
from proto.io import WriteArticle


CORENLP_URL = 'http://localhost:9000'

#corenlp_cp=""
#ram=8g

def start_corenlp_server(corenlp_cp, ram):
        FNULL = open(os.devnull, 'w')
        return subprocess.Popen([
        'java', '-mx' + ram, '-cp', corenlp_cp,
        'edu.stanford.nlp.pipeline.StanfordCoreNLPServer'],
        stdout=FNULL)


def annotate_with_corenlp(text, output_document, full):
    text = text.encode('utf-8')
    
    # Fix non-breaking spaces.
    # TODO(klopyrev): Do this in the data collection code.
    text = text.replace('\xc2\xa0', ' ')

    properties = {
        'annotators': 'tokenize,ssplit' + (',pos,ner,parse,depparse' if full else ''),
        'outputFormat': 'serialized',
        'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
        'timeout': '600000',
        'pos.maxlen': '200',
        'parse.maxlen': '200',
    }
    query = {'properties': json.dumps(properties)}
    while True:
        try:
            result = urllib2.urlopen(CORENLP_URL +
                                     '/?' + urllib.urlencode(query),
                                     data=text).read()
            break
        except urllib2.HTTPError as e:
            if e.code != httplib.INTERNAL_SERVER_ERROR:
                print >> sys.stderr, 'Received internal server error, will retry.'
                sys.stderr.flush()
                raise
            time.sleep(5.0)
    _, hdr_length = _DecodeVarint32(result, 0)
    output_document.ParseFromString(result[hdr_length:])


def wait_for_corenlp_startup(full):
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            annotate_with_corenlp('Text', CoreNLP_pb2.Document(), full)
            return
        except urllib2.URLError:
            time.sleep(0.5)

    raise Exception('CoreNLP failed to start.')


def shutdown_corenlp_server(corenlp_server_proc):
    with open('/tmp/corenlp.shutdown', 'r') as f:
        shutdown_key = f.read()
    urllib2.urlopen(CORENLP_URL + '/shutdown?key=' + shutdown_key)
    corenlp_server_proc.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corenlp-cp', default="",
                        help='Classpath to use when running CoreNLP. '
                             'The value of this flag is passed directly to '
                             'java -cp.')
    parser.add_argument('--input-json', default="D:\\ALDA Project\\train-v1.0.json",
                        help='Path to the dataset in JSON format.')
    parser.add_argument('--output-proto', default="D:\\ALDA Project\\hello.proto",
                        help='Where to output the annotated dataset as a '
                             'serialized protocol buffer.')
    parser.add_argument('--full', action='store_true',default='full',
                        help='If not specified, only tokenization and sentence '
                             'spliting is done.')
    parser.add_argument('--ram', default='8g',
                        help='Amount of RAM available for running the CoreNLP '
                             'server.')
    args = parser.parse_args()

    #corenlp_server_proc = start_corenlp_server(args.corenlp_cp, args.ram)
    try:
        wait_for_corenlp_startup(args.full)        
        with open(args.input_json, 'r') as f:
            input_data = json.loads(f.read())['data']

        with open(args.output_proto, 'wb') as f:
            articles = 0
            for input_article in input_data:
                output_article = dataset_pb2.Article()
                output_article.title = input_article['title']
            
                threads = []
                for input_paragraph in input_article['paragraphs']:
                    output_paragraph = output_article.paragraphs.add()

                    def annotate_paragraph(input_paragraph, output_paragraph):
                        annotate_with_corenlp(input_paragraph['context'],
                                              output_paragraph.context,
                                              args.full)
                        for input_qa in input_paragraph['qas']:
                            output_qa = output_paragraph.qas.add()
                            output_qa.id = input_qa['id']
                            annotate_with_corenlp(input_qa['question'],
                                                  output_qa.question,
                                                  args.full)

                            for answer in input_qa['answers']:
                                # Fix extra characters.
                                # TODO(klopyrev): Do this in the data collection code.
                                answer_text = answer['text']
                                answer_offset = answer['answer_start']
                                while len(answer_text) > 0 and answer_text[-1] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
                                    answer_text = answer_text[:-1]
                                while len(answer_text) > 0 and answer_text[0] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
                                    answer_text = answer_text[1:]
                                    answer_offset += 1

                                annotate_with_corenlp(answer_text,
                                                      output_qa.answers.add(),
                                                      args.full)
                                output_qa.answerOffsets.append(answer_offset)

                    threads.append(threading.Thread(target=annotate_paragraph,
                                                    args=[input_paragraph,
                                                          output_paragraph]))
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                WriteArticle(output_article, f)
                articles += 1
                print 'Annotated', articles, 'articles'
                sys.stdout.flush()
    finally:
        pass
        #shutdown_corenlp_server(corenlp_server_proc)
