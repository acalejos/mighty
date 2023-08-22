defmodule Mighty.Vocab do
  @type t :: %__MODULE__{
          strings: MapSet.t(),
          vectors: MapSet.t()
        }
  defstruct [:strings, :vectors]

  @callback init() :: Mighty.Vocab.t()
end
