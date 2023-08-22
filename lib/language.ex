defmodule Mighty.Language do
  @type t :: %__MODULE__{
          name: String.t(),
          tokenizer: Mighty.Tokenizer.t(),
          parser: module(),
          attrs: MapSet.t(),
          stop_words: Enum.List.t()
        }
  defstruct [:name, :tokenizer, :parser, :attrs, :stop_words, :vocab, :vectors]

  @callback init() :: Mighty.Language.t()
end
