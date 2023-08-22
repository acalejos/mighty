defmodule Mighty.Tokenizer do
  @moduledoc """
  Tokenizer Behaviour for `Mighty`.

  reference: https://spacy.io/api/tokenizer
  """
  @type t :: %__MODULE__{
          # Should have the same function signature as Regex.run/3
          prefix_run: (t(), String.t(), [term()] -> [[String.t()]]),
          suffix_run: (t(), String.t(), [term()] -> [[String.t()]]),
          # Should have the same function signature as Regex.scan/3
          infix_scan: (t(), String.t(), [term()] -> [[String.t()]]),
          # Should have the same function signature as Regex.run/3 but only match at the beginning of the string
          # (in SpaCy these uses re.match function signature)
          token_match: (t(), String.t(), [term()] -> [[String.t()]]),
          token_match: (t(), String.t(), [term()] -> [[String.t()]]),
          rules: %{String.t() => [%{integer() => String.t()}]} | nil,
          vocab: Mighty.Vocab.t()
        }
  defstruct [:prefix_run, :suffix_run, :token_match, :infix_scan, :rules, :vocab]

  @spec find_prefix(__MODULE__.t(), String.t()) :: integer() | nil
  def find_prefix(tokenizer, text) do
    tokenizer.prefix_run(tokenizer, text, [])
  end

  @spec find_suffix(__MODULE__.t(), String.t()) :: integer() | nil
  def find_suffix(tokenizer, text) do
    tokenizer.suffix_run(tokenizer, text, [])
  end

  # returns byte index and match length. If no match, returns nil
  @spec find_infix(__MODULE__.t(), String.t()) :: [{integer(), integer()}] | nil
  def find_infix(tokenizer, text) do
    tokenizer.infix_scan(tokenizer, text, [])
  end
end
