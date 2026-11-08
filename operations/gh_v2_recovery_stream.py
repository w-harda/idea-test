"""仅GH-v2历史账本的有界分项JSON读取器；不修改全局read_json。"""
import json, math
from tbps.artifacts import ArtifactError, _object

ARRAYS=frozenset(('G','H','conditions'))

def finite_float(raw):
    x=float(raw)
    if not math.isfinite(x):raise ArtifactError('nonfinite JSON number: '+raw)
    return x

def reject_constant(raw):raise ArtifactError('nonfinite JSON: '+raw)

class Reader:
    def __init__(self, file, *, chunk=65536, max_value=4*1024*1024):
        self.file=file;self.chunk=chunk;self.max_value=max_value
        self.buffer='';self.pos=0;self.offset=0;self.eof=False;self.maximum=0
        self.decoder=json.JSONDecoder(object_pairs_hook=_object,parse_constant=reject_constant,parse_float=finite_float)
    @property
    def absolute(self):return self.offset+len(self.buffer[:self.pos].encode("utf-8"))
    def more(self):
        self.offset+=len(self.buffer[:self.pos].encode("utf-8"));self.buffer=self.buffer[self.pos:];self.pos=0
        s=self.file.read(self.chunk)
        self.eof=not s;self.buffer+=s;self.maximum=max(self.maximum,len(self.buffer))
        if len(self.buffer)>self.max_value+self.chunk:raise ArtifactError('individual JSON value exceeds bounded reader limit')
    def peek(self):
        while True:
            while self.pos<len(self.buffer) and self.buffer[self.pos] in ' \t\r\n':self.pos+=1
            if self.pos<len(self.buffer):return self.buffer[self.pos]
            if self.eof:return ''
            self.more()
    def take(self,char):
        if self.peek()!=char:raise ArtifactError('invalid JSON delimiter; expected '+char)
        self.pos+=1
    def value(self):
        if not self.peek():raise ArtifactError('truncated JSON value')
        while True:
            start=self.pos
            try:v,end=self.decoder.raw_decode(self.buffer,start)
            except json.JSONDecodeError as e:
                if self.eof:raise ArtifactError('invalid/truncated JSON value') from e
                self.more();continue
            # 延迟接收边界上的标量，防止分块在1|23或1|e3之间切开。
            if (end==len(self.buffer) or self.buffer[end] not in ' \t\r\n,]}:') and not self.eof:self.more();continue
            if end-start>self.max_value:raise ArtifactError('individual JSON value exceeds bounded reader limit')
            self.pos=end;return v

def scan(path,on_item=lambda *args:None,*,arrays=ARRAYS,chunk=65536):
    """完整严格解析；三个大数组逐项回调，其余字段有界保留；拒绝尾随字节。"""
    sizes={};header={};counts={};seen=set()
    try:
        with open(path,'r',encoding='utf-8',errors='strict') as file:
            r=Reader(file,chunk=chunk);r.take('{')
            if r.peek()!='}':
                while True:
                    key=r.value()
                    if type(key) is not str or key in seen:raise ArtifactError('invalid/duplicate top-level JSON key')
                    seen.add(key);r.take(':');r.peek();start=r.absolute
                    if key in arrays:
                        r.take('[');count=0
                        if r.peek()!=']':
                            while True:
                                item=r.value()
                                if type(item) is not dict:raise ArtifactError('ledger array item must be object')
                                on_item(key,count,item);count+=1
                                delimiter=r.peek()
                                if delimiter==']':break
                                r.take(',')
                                if r.peek()==']':raise ArtifactError('trailing array comma')
                        r.take(']');counts[key]=count
                    else:header[key]=r.value()
                    sizes[key]=r.absolute-start
                    if r.peek()=='}':break
                    r.take(',')
                    if r.peek()=='}':raise ArtifactError('trailing object comma')
            r.take('}')
            if r.peek():raise ArtifactError('trailing JSON data')
            return header,counts,{'field_bytes':sizes,'max_buffer_characters':r.maximum,'chunk_characters':chunk,'max_individual_value_characters':r.max_value}
    except (UnicodeError,RecursionError) as e:raise ArtifactError('invalid JSON metadata') from e
