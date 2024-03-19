from pycv._lib._inspect import get_signature, get_params, PARAMETER
import textwrap
import re
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy


########################################################################################################################

Parameter = namedtuple('Parameter', 'name, annotation, default, summary')


########################################################################################################################

class ReadDoc(object):
    def __init__(self, docstring: str | list[str] = ""):
        if isinstance(docstring, str):
            self.docstring = textwrap.dedent(docstring).split('\n')
        else:
            self.docstring = docstring
        self._ptr = 0

    def __getitem__(self, _i) -> str:
        return self.docstring[_i]

    def __len__(self) -> int:
        return len(self.docstring)

    def reset(self) -> None:
        self._ptr = 0

    def has_next(self) -> bool:
        return self._ptr < len(self)

    def is_empty(self) -> bool:
        return not ''.join(self.docstring).strip()

    def read(self) -> str:
        if self.has_next():
            self._ptr += 1
            return self[self._ptr - 1]
        return ""

    def goto_next_non_empty(self):
        while self.has_next():
            if not self.is_empty_line(self[self._ptr]):
                break
            self._ptr += 1

    def read_to_next_empty(self) -> list[str]:
        self.goto_next_non_empty()
        l = self._ptr
        while self.has_next():
            if self.is_empty_line(self[self._ptr]):
                return self[l:self._ptr]
            self._ptr += 1
        return self[l:self._ptr]

    def peak_line(self, _i=0):
        if self._ptr + _i < len(self):
            return self[self._ptr + _i]
        return ""

    @staticmethod
    def is_empty_line(line: str) -> bool:
        return not line.strip()


class DocString(Mapping):
    sections = {
        'Signature': '',
        'Summary': [''],
        'Parameters': [],
        'Returns': [],
        'Yields': [],
        'Receives': [],
        'Raises': [],
        'Warns': [],
        'Attributes': [],
        'Methods': [],
        'Notes': [],
        'Warnings': [],
        'References': [],
        'Examples': [],
    }

    def __init__(self, docstring: str = ""):
        self.docstring = textwrap.dedent(docstring).split('\n')
        self.reader = ReadDoc(self.docstring)
        self._parse()
        self._copy = copy.deepcopy(self.sections)

    def __getitem__(self, item):
        return self.sections[item]

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self):
        return iter(self.sections)

    def __setitem__(self, section, value):
        if section not in self.sections:
            raise KeyError(f'unknown section {section}')
        self.sections[section] = value

    def _add_summery(self):
        return self['Summary'] + [""]

    def _add_header(self, header, sep="-") -> list:
        return [header, sep * len(header)]

    def _add_lines_shift(self, lines, shift=4) -> list:
        out = []
        for line in lines:
            out.append(" " * shift + line)
        return out

    def _add_section(self, section) -> list:
        out = []
        if self[section]:
            out += self._add_header(section)
            for param in self[section]:
                p = param.name
                if param.annotation:
                    p += f": {' | '.join(param.annotation)}"
                if param.default:
                    p += f" = {param.default}"
                if not param.annotation or not param.default:
                    p += ":"
                out += [p]
                if param.summary:
                    out += self._add_lines_shift(param.summary)
            out += [""]
        return out

    def __str__(self):
        out = []
        out += [self['Signature']]
        out += self._add_summery()

        for section in ('Parameters', 'Returns', 'Yields', 'Receives', 'Raises', 'Warns', 'Attributes', 'Methods'):
            out += self._add_section(section)

        for section in ('Warnings', 'Notes', 'References', 'Examples'):
            if self[section]:
                out += self._add_header(section)
                out.extend(self[section])
                out += [""]

        return '\n'.join(out)

    def _is_reader_in_section(self):
        self.reader.goto_next_non_empty()
        if not self.reader.has_next():
            return False
        line1 = self.reader.peak_line().strip()
        line2 = self.reader.peak_line(1).strip()
        if line2 == "-" * len(line1):
            return True
        return False

    def _parse_summery(self):
        if self._is_reader_in_section():
            return

        summary = self.reader.read_to_next_empty()
        as_str = ''.join([s.strip() for s in summary]).strip()
        is_signature = False

        if "(" in as_str and ")" in as_str:
            pattern = re.split(r'[\(\)]', as_str)
            if len(pattern) >= 3:
                prefix = re.split(r'[\s\.]', pattern[0])

                if prefix[0] in ('Class', 'Function') and pattern[0].count(".") == len(prefix) - 2:
                    self['Signature'] = as_str
                    is_signature = True
        if not is_signature:
            self['Summary'].extend(s for s in summary)

        while not self._is_reader_in_section():
            self['Summary'].extend(s for s in self.reader.read_to_next_empty())

    def _parse_sections(self):
        out = []
        while self.reader.has_next():
            section = self.reader.read_to_next_empty()
            while not self._is_reader_in_section() and self.reader.has_next():
                if not self.reader.peak_line(-1).strip():
                    section.append("")
                section += self.reader.read_to_next_empty()
            if len(section) < 2:
                return out
            header = section[0].strip()
            out.append((header, section[2:]))
        return out

    def _parse_parameters(self, name: str, contents: list):
        if name not in self.sections or not contents:
            return

        curr_summery = []

        for content in contents:
            if content.strip() and (len(content.lstrip()) == len(content)):
                annotation = [s.strip() for s in re.split("[:|=]", content)]
                param_name = annotation.pop(0)
                default = ""
                if content.find('=') != -1:
                    default = annotation.pop()

                curr_summery = []
                curr_param = Parameter(param_name, annotation, default, curr_summery)
                self[name].append(curr_param)
            else:
                curr_summery.append(content.strip())

    def _parse(self):
        self.reader.reset()
        self._parse_summery()

        for name, contents in self._parse_sections():
            self._parse_parameters(name, contents)


########################################################################################################################

def function_doc_from_inspect(doc: DocString, func: Callable, qualname: str = "", module: str = ""):
    def _parse_type_str(txt: str) -> str:
        for a in [s.strip() for s in re.split("[<'>]", txt)]:
            if not a or a == 'class':
                continue
            return a
        return ""

    def _parse_inspect_parameter(parameter: PARAMETER) -> Parameter:
        _name = parameter.name

        if parameter.annotation is parameter.empty:
            _annotation = []
        else:
            _annotation = [p.strip() for p in str(parameter.annotation).split("|")]

        if len(_annotation) == 1:
            _annotation = [_parse_type_str(_annotation[0])]

        if parameter.default is parameter.empty:
            default = ""
        else:
            default = str(parameter.default)

        if parameter.kind == parameter.VAR_POSITIONAL:
            _annotation = ['tuple']
            _name = '*' + _name
        elif parameter.kind == parameter.VAR_KEYWORD:
            _annotation = ['dict']
            _name = '**' + _name
        return Parameter(_name, _annotation, default, [])

    signature = get_signature(func)

    name = module or func.__module__
    if qualname:
        name += f".{qualname}"

    doc['Signature'] = f'Function {name}.{func.__name__}{str(signature)}'

    for param in get_params(signature):
        doc['Parameters'].append(_parse_inspect_parameter(param))

    if signature.return_annotation is not signature.empty:
        annotation = str(signature.return_annotation)
        if annotation.find("[") != -1:
            annotation = annotation.split("[")[1][:-1]
        else:
            annotation = _parse_type_str(annotation)

        for i, param in enumerate(annotation.split(",")):
            sub_annotation = [a.strip() for a in re.split("[|]", param)]
            doc['Returns'].append(Parameter(f"output{i + 1}", sub_annotation, "", []))


########################################################################################################################

