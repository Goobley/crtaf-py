import ruamel.yaml
from ruamel.yaml.comments import Format, format_attrib
from test_model import Data
from crtaf.core_types import Atom
from io import StringIO

if __name__ == "__main__":
    atom = Atom.model_validate(Data)
    data = atom.model_dump()
    yaml = ruamel.yaml.YAML(typ='rt')
    yaml.compact(seq_seq=True, seq_map=True)
    yaml.sort_base_mapping_type_on_output = False
    yaml.default_flow_style = False

    stream = StringIO()
    yaml.dump(data, stream=stream)
    print(stream.getvalue())

    print('----\n\n')

    # ser, rep, em = yaml.get_serializer_representer_emitter(stream, False)
    # dd = rep.represent(data)
    # dd = yaml.Xdump_all(data, stream)
    dd = yaml.representer.represent_data(data)

    # print(dd)

    rep = yaml.load(stream.getvalue())
    stream = StringIO()
    yaml.dump(rep, stream)
    print(stream.getvalue())

    rep['radiative_bound_free'][0]['value'].fa.set_flow_style()
    stream = StringIO()
    yaml.dump(rep, stream)
    print(stream.getvalue())